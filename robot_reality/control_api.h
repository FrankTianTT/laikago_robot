/**
 * @file control_api.h
 * 
 * This file defines the Sensor class that is consistent with the sensor
 * defined in laikago_py repository. 
 * 
 * The sensor including: motor angle, velocity, torque sensor, IMU sensor,
 * toe force sensor. The python script needs to record last action which is 
 * used as a last action sensor when training the network.
 * 
 * @author Song Lei 2020.9.16
 */

#ifndef CONTROL_API_H_
#define CONTROL_API_H_

#include "laikago_sdk/laikago_sdk.hpp"

using namespace laikago;

/**
 * Use sensor to get the observation
 */

#define N_MOTOR 12

typedef struct MotorAngleState {
  float angle[N_MOTOR+1];
} MotorAngleState;

typedef struct MotorVelocityState {
  float vel[N_MOTOR+1];
} MotorVelocityState;

typedef struct MotorTorqueState {
  float torque[N_MOTOR+1];
} MotorTorqueState;

typedef struct IMUState {
  float gyroscope[3];
  float rpy[3];
} IMUState;

typedef struct ToeForceState {
  float foot_force[4];
} ToeForceState;

class Sensor {
 private:
  MotorAngleState motor_angle_state;
  MotorVelocityState motor_velocity_state;
  IMUState imu_state;
  ToeForceState toe_force_state;

 public:
  Sensor() = default;
  void UpdateMotorAngle(const LowState state);
  void UpdateMotorVelocity(const LowState state);
  void UpdateIMU(const LowState state);
  void UpdateToeForce(const LowState state);
  void UpdateSensor(const LowState state);
  void ConvertSensor2Obs(float *obs) const;
};

/**
 * Set the action
 */

extern void SetAction(LowCmd &cmd, const MotorAngleState pos);

extern void SetAction(LowCmd &cmd, const MotorVelocityState vel);
extern void SetAction(LowCmd &cmd, const MotorTorqueState tor);

#endif // CONTROL_API_H_
