/**
 * @file client.cpp
 * 
 * Control laikago with time shift
 *
 * @author Song Lei
 */

#include "control_api.h"
#include "laikago_sdk/laikago_sdk.hpp"
#include <iostream>
#include <stdlib.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <assert.h>
#include <time.h>

#define PORT 1235
#define OBS_SIZE 34
#define ACTION_SIZE 12
int torque_limit[] = {1, 1, 1, 1};

using namespace laikago;
using std::cout;
using std::endl;

// PD parameters
const float ABDUCTION_P_GAIN = 50.0;
const float ABDUCTION_D_GAIN = 1.0;
const float HIP_P_GAIN = 50.0;
const float HIP_D_GAIN = 2.0;
const float KNEE_P_GAIN = 50.0;
const float KNEE_D_GAIN = 2.0;
float kp[ACTION_SIZE] = {ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                         ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                         ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                         ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN};
float kd[ACTION_SIZE] = {ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                         ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                         ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                         ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN};

const float ABDUCTION_MAX_POS_CHANGE = 0.175;  // 10 degree
const float HIP_MAX_POS_CHANGE = 0.175;
const float KNEE_MAX_POS_CHANGE = 0.175;
float max_pos_change[ACTION_SIZE] = {ABDUCTION_MAX_POS_CHANGE, HIP_MAX_POS_CHANGE, KNEE_MAX_POS_CHANGE,
                                     ABDUCTION_MAX_POS_CHANGE, HIP_MAX_POS_CHANGE, KNEE_MAX_POS_CHANGE,
                                     ABDUCTION_MAX_POS_CHANGE, HIP_MAX_POS_CHANGE, KNEE_MAX_POS_CHANGE,
                                     ABDUCTION_MAX_POS_CHANGE, HIP_MAX_POS_CHANGE, KNEE_MAX_POS_CHANGE};

// standard variables when controlling laikago on low level
static long motiontime = 0;
LowCmd cmd = {0};
LowState state = {0};
Control control(LOWLEVEL);
UDP udp(LOW_CMD_LENGTH, LOW_STATE_LENGTH);
Sensor sensor;

// Process the action
float obs[OBS_SIZE];
float last_pos[ACTION_SIZE], target_pos[ACTION_SIZE];
float cur_torque[ACTION_SIZE] = {0};
clock_t cur = 0, large_step_pre = 0, small_step_pre = 0;
int cur_smooth_step = 0;
int send_obs_flag = 1;
const int MAX_REPEAT_STEP = 10;
const float LARGE_STEP = 0.020, SMALL_STEP = 0.001;

// socket
int client_fd = 0;
struct sockaddr_in server_addr;

void PrintMotorState(const LowState state) {
  cout << "-------------------------------------------" << endl;
  for(int i = 1; i <= N_MOTOR; i++) {
    cout << "position: " << state.motorState[i].position << ", "
         << "velocity: " << state.motorState[i].velocity << ", "
         << "torque: "   << state.motorState[i].torque   << '.'  << endl;
  }
  cout << "-------------------------------------------" << endl;
}

void UDPRecv() {
  udp.Recv();
}

void SetCmdTorque(LowCmd &cmd, int idx, float torque) {
  cmd.motorCmd[idx].position = PosStopF;
  cmd.motorCmd[idx].velocity = VelStopF;
  cmd.motorCmd[idx].positionStiffness = 0;
  cmd.motorCmd[idx].velocityStiffness = 0;
  cmd.motorCmd[idx].torque = torque;
}

float clip(float x, float min_x, float max_x) {
  assert(min_x <= max_x);
  float cliped_x = x;
  if(x > max_x) {
    cliped_x = max_x;
  } else if (x < min_x) {
    cliped_x = min_x;
  }
  return cliped_x;
}

float smooth_action(float old_action, float target_action, int step) {
  // linear interpolation
  float lerp = (float)(step + 1) / MAX_REPEAT_STEP;
  float cur_action = old_action + lerp * (target_action - old_action);
  return cur_action;
}

/**
 * 1. Receive a action from python per LARGE_STEP.
 * 2. Smooth and clip the action per SMALL_STEP.
 * 3. Convert to torques using PD.
 */

void RobotControl() {
  motiontime ++; 
  cur = clock();

  udp.GetState(state);
  // PrintMotorState(state);
  sensor.UpdateSensor(state);
  sensor.ConvertSensor2Obs(obs);


  if(motiontime == 1) {
    cout << "Get init position" << endl;
    int len = recv(client_fd, (char*)target_pos, ACTION_SIZE*sizeof(float), 0);  // 阻塞接受初始pos
    assert(len == ACTION_SIZE*sizeof(float));
    small_step_pre = large_step_pre = clock();
    for(int i = 0; i < N_MOTOR; i++)
      last_pos[i] = target_pos[i];
  }

  if(send_obs_flag) {
    send(client_fd, (char*)obs, OBS_SIZE*sizeof(float), 0);
    send_obs_flag = 0;
  }

  // 每SMALL_STEP更新一次torque
  if(((double)(cur - small_step_pre)) / CLOCKS_PER_SEC >= SMALL_STEP) {
    // small_step_pre = clock();
    float next_action = 0;
    small_step_pre += CLOCKS_PER_SEC * SMALL_STEP;
    for(int i = 0; i < N_MOTOR; i++) {
      // 注意state.motorState下标是从1开始计算的
      next_action = smooth_action(last_pos[i], target_pos[i], cur_smooth_step);
      float pos_change = state.motorState[i+1].position - next_action;
      pos_change = clip(pos_change, - max_pos_change[i], max_pos_change[i]);
      cur_torque[i] = -1 * kp[i] * pos_change - kd[i] * (state.motorState[i+1].velocity - 0);
    }

    cur_smooth_step += 1;
  }

  // 每LARGE_STEP更新一次action
  if(((double)(cur - large_step_pre)) / CLOCKS_PER_SEC >= LARGE_STEP) {
    // large_step_pre = clock();
    for(int i = 0; i < N_MOTOR; i++)
      last_pos[i] = target_pos[i];

    large_step_pre += CLOCKS_PER_SEC * LARGE_STEP;
    int len = recv(client_fd, (char*)target_pos, ACTION_SIZE*sizeof(float), MSG_DONTWAIT);
    // int len = recv(client_fd, (char*)target_pos, ACTION_SIZE*sizeof(float), 0);
    // cout << "length: " << len << endl;
    cout << "len:" << len << endl;
    assert(len == ACTION_SIZE*sizeof(float));

    float tmp = 0;
    if(recv(client_fd, &tmp, sizeof(float), MSG_DONTWAIT) != -1)  // 检查socket缓冲区是否为空
      assert(0);

    send_obs_flag = 1;
    // cout << "Smooth count: " << cur_smooth_step << endl;
    cur_smooth_step = 0;
  }

  // cout << "Get target_pos: ";
  // for(int i = 0; i < ACTION_SIZE; i++)
  //   cout << target_pos[i] << " ";
  // cout << endl;

  for(int i = 0; i < 4; i++) {
    if(state.footForce[i] > 0) {
      torque_limit[i] = 1+0.25*state.footForce[i];
      if(torque_limit[i] > 10)
        torque_limit[i] = 10;
    } else {
      torque_limit[i] = 1;
    }
  }
  cout << torque_limit[0] << endl;

  for(int i = 0; i < N_MOTOR; i++) {
    if(cur_torque[i] < -torque_limit[i/3])
      cur_torque[i] = -torque_limit[i/3];
    if(cur_torque[i] > torque_limit[i/3])
      cur_torque[i] = torque_limit[i/3];
  }

  MotorTorqueState tor;
  for(int i = 1; i <= N_MOTOR; i++)
    tor.torque[i] = cur_torque[i-1];
  // SetAction(cmd, tor);

  SetCmdTorque(cmd, 1, tor.torque[1]);
  SetCmdTorque(cmd, 2, tor.torque[2]);
  SetCmdTorque(cmd, 3, tor.torque[3]);
  SetCmdTorque(cmd, 4, tor.torque[4]);
  SetCmdTorque(cmd, 5, tor.torque[5]);
  SetCmdTorque(cmd, 6, tor.torque[6]);
  SetCmdTorque(cmd, 7, tor.torque[7]);
  SetCmdTorque(cmd, 8, tor.torque[8]);
  SetCmdTorque(cmd, 9, tor.torque[9]);
  SetCmdTorque(cmd, 10, tor.torque[10]);
  SetCmdTorque(cmd, 11, tor.torque[11]);
  SetCmdTorque(cmd, 12, tor.torque[12]);


  control.JointLimit(cmd);
  control.PowerLimit(cmd, state, 1);

  udp.Send(cmd);
}

int main() {
  cout << "Control level is set to LOW-level" << endl <<
          "You should remember to sudo" << endl;
  std::cin.ignore();

  assert((client_fd = socket(AF_INET, SOCK_STREAM, 0)) != -1);
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(PORT);
  server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  assert((connect(client_fd, (struct sockaddr*)&server_addr, sizeof(server_addr))) >= 0);

  control.InitCmdData(cmd);
  control.loop.RegistFunc("UDP/Send", RobotControl);
  control.loop.RegistFunc("UDP/Recv", UDPRecv);
  control.loop.Start();

  return 0;
}
