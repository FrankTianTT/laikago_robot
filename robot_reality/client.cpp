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
const float ABDUCTION_P_GAIN = 220.0;
const float ABDUCTION_D_GAIN = 0.3;
const float HIP_P_GAIN = 220.0;
const float HIP_D_GAIN = 2.0;
const float KNEE_P_GAIN = 220.0;
const float KNEE_D_GAIN = 2.0;
float kp[ACTION_SIZE] = {ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                         ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                         ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                         ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN};
float kd[ACTION_SIZE] = {ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                         ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                         ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                         ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN};

static long motiontime=0;
LowCmd cmd = {0};
LowState state = {0};
Control control(LOWLEVEL);
UDP udp(LOW_CMD_LENGTH, LOW_STATE_LENGTH);
Sensor sensor;

float obs[OBS_SIZE];
float pos[ACTION_SIZE];
float cur_torque[ACTION_SIZE];

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

void RobotControl() {
  motiontime ++;
  cur = clock();

  // Get obs from current state
  udp.GetState(state);
  // PrintMotorState(state);
  sensor.UpdateSensor(state);
  sensor.ConvertSensor2Obs(obs);

  if(send_obs_flag) {
    send(client_fd, (char*)obs, OBS_SIZE*sizeof(float), 0);
    send_obs_flag = 0;
  }

  // 每2ms更新一次torque
  if(((double)(cur - small_step_pre)) / CLOCKS_PER_SEC >= 0.002) {
    small_step_pre = clock();
    for(int i = 0; i < N_MOTOR; i++) {
      // 注意state.motorState下标是从1开始计算的
      cur_torque[i] = -1 * (kp[i] * (state.motorState[i+1].position - pos[i])) - kd[i] * (state.motorState[i+1].velocity - 0);
    }
  }

  // 每20ms更新一次action
  if(((double)(cur - large_step_pre)) / CLOCKS_PER_SEC >= 0.020) {
    large_step_pre = clock();
    int len = recv(client_fd, (char*)pos, ACTION_SIZE*sizeof(float), MSG_DONTWAIT);
    assert(len == ACTION_SIZE*sizeof(float));

    float tmp = 0;
    if(recv(client_fd, &tmp, sizeof(float), MSG_DONTWAIT) != -1)  // 检查socket缓冲区是否为空
      assert(0);

    send_obs_flag = 1;
  }

  // cout << "Get action: ";
  // for(int i = 0; i < ACTION_SIZE; i++)
  //   cout << action[i] << " ";
  // cout << endl;

  MotorTorqueState tor;
  for(int i = 1; i <= N_MOTOR; i++)
    tor.torque[i] = cur_torque[i-1];
  // SetAction(cmd, tor);

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
