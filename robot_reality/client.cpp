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

#define PORT 12345
#define OBS_SIZE 34
#define ACTION_SIZE 12

using namespace laikago;
using std::cout;
using std::endl;

// PD parameters
const float ABDUCTION_P_GAIN = 25.0;
const float ABDUCTION_D_GAIN = 1;
const float HIP_P_GAIN = 20.0;
const float HIP_D_GAIN = 0.5;
const float KNEE_P_GAIN = 25.0;
const float KNEE_D_GAIN = 0.5;
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

static long motiontime=0;
LowCmd cmd = {0};
LowState state = {0};
Control control(LOWLEVEL);
UDP udp(LOW_CMD_LENGTH, LOW_STATE_LENGTH);
Sensor sensor;

float obs[OBS_SIZE];
float pos[ACTION_SIZE];
float cur_torque[ACTION_SIZE] = {0};

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

clock_t cur = 0, large_step_pre = 0, small_step_pre = 0;
int send_obs_flag = 1;
const float LARGE_STEP = 0.020, SMALL_STEP = 0.002;

void SetCmdTorque(LowCmd &cmd, int idx, float torque) {
  cmd.motorCmd[idx].position = PosStopF;
  cmd.motorCmd[idx].velocity = VelStopF;
  cmd.motorCmd[idx].positionStiffness = 0;
  cmd.motorCmd[idx].velocityStiffness = 0;
  cmd.motorCmd[idx].torque = torque;
}

void RobotControl() {
  motiontime ++; 
  cur = clock();

  // Get obs from current state
  udp.GetState(state);
  // PrintMotorState(state);
  sensor.UpdateSensor(state);
  sensor.ConvertSensor2Obs(obs);

  if(motiontime == 1) {
    cout << "Get init position" << endl;
    int len = recv(client_fd, (char*)pos, ACTION_SIZE*sizeof(float), 0);  // 阻塞接受初始pos
    assert(len == ACTION_SIZE*sizeof(float));
    small_step_pre = large_step_pre = clock();
  }

  if(send_obs_flag) {
    send(client_fd, (char*)obs, OBS_SIZE*sizeof(float), 0);
    send_obs_flag = 0;
  }

  // 每SMALL_STEP更新一次torque
  if(((double)(cur - small_step_pre)) / CLOCKS_PER_SEC >= SMALL_STEP) {
    // small_step_pre = clock();
    small_step_pre += CLOCKS_PER_SEC * SMALL_STEP;
    for(int i = 0; i < N_MOTOR; i++) {
      // 注意state.motorState下标是从1开始计算的
      float pos_change = state.motorState[i+1].position - pos[i];
      cout << "pos_change: " << pos_change << endl;
      if(pos_change > max_pos_change[i]) {
        pos_change = max_pos_change[i];
      } else if (pos_change < - max_pos_change[i]) {
        pos_change = - max_pos_change[i];
      }
      cout << "after clip: pos_change: " << pos_change << endl;
      cur_torque[i] = -1 * kp[i] * pos_change - kd[i] * (state.motorState[i+1].velocity - 0);
    }
  }

  // 每LARGE_STEP更新一次action
  if(((double)(cur - large_step_pre)) / CLOCKS_PER_SEC >= LARGE_STEP) {
    // large_step_pre = clock();
    large_step_pre += CLOCKS_PER_SEC * LARGE_STEP;
    int len = recv(client_fd, (char*)pos, ACTION_SIZE*sizeof(float), MSG_DONTWAIT);
    // cout << "length: " << len << endl;
    assert(len == ACTION_SIZE*sizeof(float));

    float tmp = 0;
    if(recv(client_fd, &tmp, sizeof(float), MSG_DONTWAIT) != -1)  // 检查socket缓冲区是否为空
      assert(0);

    send_obs_flag = 1;
  }

  cout << "Get pos: ";
  for(int i = 0; i < ACTION_SIZE; i++)
    cout << pos[i] << " ";
  cout << endl;

  MotorTorqueState tor;
  for(int i = 1; i <= N_MOTOR; i++)
    tor.torque[i] = cur_torque[i-1];
  // SetAction(cmd, tor);

  SetCmdTorque(cmd, 1, tor.torque[1]);
  SetCmdTorque(cmd, 2, tor.torque[2]);
  SetCmdTorque(cmd, 3, tor.torque[3]);
  // SetCmdTorque(cmd, 4, tor.torque[4]);
  // SetCmdTorque(cmd, 5, tor.torque[5]);
  // SetCmdTorque(cmd, 6, tor.torque[6]);
  // SetCmdTorque(cmd, 7, tor.torque[7]);
  // SetCmdTorque(cmd, 8, tor.torque[8]);
  // SetCmdTorque(cmd, 9, tor.torque[9]);
  // SetCmdTorque(cmd, 10, tor.torque[10]);
  // SetCmdTorque(cmd, 11, tor.torque[11]);
  // SetCmdTorque(cmd, 12, tor.torque[12]);


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
  server_addr.sin_addr.s_addr = inet_addr("127.0.0.5");
  assert((connect(client_fd, (struct sockaddr*)&server_addr, sizeof(server_addr))) >= 0);

  control.InitCmdData(cmd);
  control.loop.RegistFunc("UDP/Send", RobotControl);
  control.loop.RegistFunc("UDP/Recv", UDPRecv);
  control.loop.Start();

  return 0;
}
