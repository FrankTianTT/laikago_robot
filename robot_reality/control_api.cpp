/**
 * @file control_api.cpp
 * 
 * This file provides APIs to Update and set state when implementing RL
 * 
 * @author Song Lei 2020.9.16
 */

#include "control_api.h"
#include "laikago_sdk/laikago_sdk.hpp"
#include <cstring>
using laikago::LowState;
using laikago::LowCmd;
using laikago::VelStopF;
using laikago::PosStopF;

/**
 * Update all sensors
 */

void Sensor::UpdateMotorAngle(const LowState state) {
  for(int i = 1; i <= N_MOTOR; i++) 
    motor_angle_state.angle[i] = state.motorState[i].position;
}

void Sensor::UpdateMotorVelocity(const LowState state) {
  for(int i = 1; i <= N_MOTOR; i++) 
    motor_velocity_state.vel[i] = state.motorState[i].velocity;
}

void Sensor::UpdateIMU(const LowState state) {
  for(int i = 0; i < 3; i++)
    imu_state.gyroscope[i] = state.imu.gyroscope[i];
  for(int i = 0; i < 3; i++)
    imu_state.rpy[i] = state.imu.rpy[i];
}

void Sensor::UpdateToeForce(const LowState state) {
  for(int i = 0; i < 4; i++)
    toe_force_state.foot_force[i] = state.footForce[i];
}

void Sensor::UpdateSensor(const LowState state) {
  UpdateMotorAngle(state);
  UpdateMotorVelocity(state);
  UpdateIMU(state);
  UpdateToeForce(state);
}

/**
 * Convert the sensor data to a float array. The size of obs needs to 
 * be consistent with the size of sensor.
 */

void Sensor::ConvertSensor2Obs(float *obs) const {
  // TODO: Adjust the sequence of sensor, and test the function
  for(int i = 1; i <= N_MOTOR; i++)
    obs[i-1] = motor_angle_state.angle[i];
  for(int i = 1; i <= N_MOTOR; i++)
    obs[12+i-1] = motor_velocity_state.vel[i];
  for(int i = 0; i < 3; i++)
    obs[24+i] = imu_state.rpy[i];
  for(int i = 0; i < 3; i++)
    obs[27+i] = imu_state.gyroscope[i];
  for(int i = 0; i < 4; i++)
    obs[30+i] = toe_force_state.foot_force[i];
}

/**
 * Set action, control the robot in different mode, including position
 * mode, velocity mode, torque mode according to the second parameters.
 */

void SetAction(LowCmd &cmd, const MotorAngleState pos) {
  for(int i = 1; i <= N_MOTOR; i++) {
    cmd.motorCmd[i].position = pos.angle[i];
    cmd.motorCmd[i].velocity = VelStopF;
    if(i % 3 == 1)
      cmd.motorCmd[i].positionStiffness = 1;
    else
      cmd.motorCmd[i].positionStiffness = 0.1;
    cmd.motorCmd[i].velocityStiffness = 0.02;
    cmd.motorCmd[i].torque = 0;
  }

  cmd.motorCmd[FR_0].torque = -0.65f;
  cmd.motorCmd[FL_0].torque = +0.65f;
  cmd.motorCmd[RR_0].torque = -0.65f;
  cmd.motorCmd[RL_0].torque = +0.65f;
}

void SetAction(LowCmd &cmd, const MotorVelocityState vel) {
  for(int i = 1; i <= N_MOTOR; i++) {
    cmd.motorCmd[i].position = PosStopF;
    cmd.motorCmd[i].velocity = vel.vel[i];
    cmd.motorCmd[i].positionStiffness = 0;
    cmd.motorCmd[i].velocityStiffness = 0.02;
    cmd.motorCmd[i].torque = 0;
  }
}

void SetAction(LowCmd &cmd, const MotorTorqueState tor) {
  for(int i = 1; i <= N_MOTOR; i++) {
    cmd.motorCmd[i].position = PosStopF;
    cmd.motorCmd[i].velocity = VelStopF;
    cmd.motorCmd[i].positionStiffness = 0;
    cmd.motorCmd[i].velocityStiffness = 0;
    cmd.motorCmd[i].torque = tor.torque[i];
  }
}
