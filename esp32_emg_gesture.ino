
// ESP32-S3 EMG Gesture Recognition
// Using Gravity Analog EMG Sensor by OYMotion
#include <Arduino.h>
const int EMG_PIN = 34;  // Analog input pin
const int WINDOW_SIZE = 50;
const int STEP_SIZE = 10;
const int SAMPLING_RATE = 1000;  // Hz
float emg_buffer[WINDOW_SIZE];
int buffer_index = 0;
// Feature extraction functions
float calculate_mean(float* data, int len) {
    float sum = 0;
    for(int i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum / len;
}
float calculate_std(float* data, int len) {
    float mean = calculate_mean(data, len);
    float sum = 0;
    for(int i = 0; i < len; i++) {
        sum += pow(data[i] - mean, 2);
    }
    return sqrt(sum / len);
}
float calculate_rms(float* data, int len) {
    float sum = 0;
    for(int i = 0; i < len; i++) {
        sum += pow(data[i], 2);
    }
    return sqrt(sum / len);
}
int calculate_zcr(float* data, int len) {
    int count = 0;
    for(int i = 1; i < len; i++) {
        if((data[i] >= 0 && data[i-1] < 0) || 
           (data[i] < 0 && data[i-1] >= 0)) {
            count++;
        }
    }
    return count;
}
void setup() {
    Serial.begin(115200);
    pinMode(EMG_PIN, INPUT);
}
void loop() {
    // Read EMG value
    int raw_value = analogRead(EMG_PIN);
    float emg_value = (raw_value / 4095.0) * 4000;  // Convert to 0-4000 range
    
    // Add to buffer
    emg_buffer[buffer_index] = emg_value;
    buffer_index = (buffer_index + 1) % WINDOW_SIZE;
    
    // Process when buffer is full
    if(buffer_index == 0) {
        // Extract features
        float mean = calculate_mean(emg_buffer, WINDOW_SIZE);
        float std = calculate_std(emg_buffer, WINDOW_SIZE);
        float rms = calculate_rms(emg_buffer, WINDOW_SIZE);
        int zcr = calculate_zcr(emg_buffer, WINDOW_SIZE);
        
        // Send features to serial for processing
        Serial.print("FEATURES:");
        Serial.print(mean); Serial.print(",");
        Serial.print(std); Serial.print(",");
        Serial.print(rms); Serial.print(",");
        Serial.print(zcr);
        Serial.println();
    }
    
    delay(1000 / SAMPLING_RATE);  // Maintain sampling rate
}
