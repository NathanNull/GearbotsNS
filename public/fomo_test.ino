#include <ei-Nathan-fomo-1redleft-2whiteright-3notefast_inferencing.h>
#include "edge-impulse-advanced-v2.h"
#include <Servo.h>
#include "mbed.h"
#include "rtos.h"
using namespace rtos;

#define CUTOUT_COLS EI_CLASSIFIER_INPUT_WIDTH
#define CUTOUT_ROWS EI_CLASSIFIER_INPUT_HEIGHT
const int cutout_row_start = (EI_CAMERA_RAW_FRAME_BUFFER_ROWS - CUTOUT_ROWS) / 2;
const int cutout_col_start = (EI_CAMERA_RAW_FRAME_BUFFER_COLS - CUTOUT_COLS) / 2;

// Thread globals
Thread thread_01;
const int THREAD_SLEEP = 10;

// Servo globals
Servo servo_d2;
const int SERVO_STRAIGHT = 90;
const int SERVO_TURN = 5;
const int SERVO_MIN = SERVO_STRAIGHT - SERVO_TURN;
const int SERVO_MAX = SERVO_STRAIGHT + SERVO_TURN;
int servo_now = SERVO_STRAIGHT;

// ML const
const float FOMO_CUTOFF = 0.85;

// PWM globals
const int SPEED = 35;
int pwm_now = 0;

// Allowed positions for objects on camera
const int MIDDLE_X = 42;
const int ALLOWED_X_DIFF = 15;
const int MIN_X = MIDDLE_X - ALLOWED_X_DIFF;
const int MAX_X = MIDDLE_X + ALLOWED_X_DIFF;

// Current D5 value
int d5_now = 0;

void LEDBlueThread() {
  while (true) {
    analogWrite(D5, d5_now);
    servo_d2.write(servo_now);
    ThisThread::sleep_for(THREAD_SLEEP);
  }
}

void setup() {
  thread_01.start(LEDBlueThread);
  Serial.begin(115200);

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  servo_d2.attach(D2);

  pinMode(D3, OUTPUT);
  pinMode(D5, OUTPUT);
  digitalWrite(D3, 1);

#ifdef EI_CAMERA_FRAME_BUFFER_SDRAM
  SDRAM.begin(SDRAM_START_ADDRESS);
#endif

  if (ei_camera_init()) {
    Serial.println("Failed to init camera");
  }
  else {
    Serial.println("Camera initialized");
  }

  for (size_t ix = 0; ix < ei_dsp_blocks_size; ix++) {
    ei_model_dsp_t block = ei_dsp_blocks[ix];
    if (block.extract_fn == &extract_image_features) {
      ei_dsp_config_image_t config = *((ei_dsp_config_image_t*)block.config);
      int16_t channel_count = strcmp(config.channels, "Grayscale") == 0 ? 1 : 3;
      if (channel_count == 3) {
        Serial.println("WARN: You've deployed a color model, but the Arduino Portenta H7 only has a monochrome image sensor. Set your DSP block to 'Grayscale' for best performance.");
        break;  // only print this once
      }
    }
  }
}

void loop() {
  if (ei_sleep(0) != EI_IMPULSE_OK) {
    return;
  }

  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
  signal.get_data = &ei_camera_cutout_get_data;

  if (!ei_camera_capture((size_t) EI_CLASSIFIER_INPUT_WIDTH, (size_t) EI_CLASSIFIER_INPUT_HEIGHT, NULL))
  {
    Serial.println("Failed to capture image");
    return;
  }

  ei_impulse_result_t result =  {0};

  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
  if (err != EI_IMPULSE_OK) {
    Serial.println("ERR: Classifier failed, error #"+String(err));
    return;
  }

  bool bb_found = result.bounding_boxes[0].value > 0;

  ei_impulse_result_bounding_box_t boxes[10];
  for (size_t ix=0; ix<10; ix++) {
    auto bb = result.bounding_boxes[ix];
    if (bb.value == 0) {
      continue;
    }

    // ei_printf("    %s (", bb.label);
    // ei_printf_float(bb.value);
    // ei_printf(") [ x: %u, y: %u, width: %u, height: %u ]\n", bb.x, bb.y, bb.width, bb.height);

    if ((float) bb.value > FOMO_CUTOFF) {
      ei_printf("type: %u, confidence: %u, GOOD", bb.label, bb.value);
      boxes[ix] = bb;
    } else {
      ei_printf("type: %u, confidence: %u, BAD", bb.label, bb.value);
    }
  }

  if (!bb_found) {
    Serial.println("Nothing found");
  }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
  Serial.println(", Anomaly: "+String(result.anomaly, 5));
#endif

  if (bb_found) {
    ei_impulse_result_bounding_box_t best = ei_impulse_result_bounding_box_t();
    auto base = best; // Jank thingy to detect if no valid objects found
    for (auto bb: boxes) {
      if (bb.x > MIN_X && bb.x < MAX_X) {
        // Y greater -> lower? Higher? IDK.
        if (best.y < bb.y) {
          best = bb;
        }
      }
    }

    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);

    if (best.value == base.value) {
      Serial.println("Nothing within bounds.");
      pwm_now = 0;
      return;
    }
    else {
      pwm_now = SPEED;
    }

    if (best.label == "1rl") {
      servo_now = SERVO_MAX;
      digitalWrite(LEDR, LOW);
    }
    else if (best.label == "2wr"){
      servo_now = SERVO_MIN;
      digitalWrite(LEDG, LOW);
    }
    else if (best.label == "3nf"){
      servo_now = SERVO_STRAIGHT;
      digitalWrite(LEDB, LOW);
    }
    else {
      Serial.println(best.label);
    }
  }
}