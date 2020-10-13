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

static long motiontime=0;
LowCmd cmd = {0};
LowState state = {0};
Control control(LOWLEVEL);
UDP udp(LOW_CMD_LENGTH, LOW_STATE_LENGTH);
Sensor sensor;

float obs[OBS_SIZE];
float action[ACTION_SIZE];

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

clock_t cur, pre;

void RobotControl() {
  cur = clock();
  printf("%f\n",  (double)(cur - pre)/CLOCKS_PER_SEC);

  motiontime ++;
  udp.GetState(state);
  PrintMotorState(state);

  sensor.UpdateSensor(state);

  sensor.ConvertSensor2Obs(obs);

  send(client_fd, (char*)obs, OBS_SIZE*sizeof(float), 0);

  int len = recv(client_fd, (char*)action, ACTION_SIZE*sizeof(float), 0);
  assert(len == ACTION_SIZE*sizeof(float));

  // cout << "Get action: ";
  // for(int i = 0; i < ACTION_SIZE; i++)
  //   cout << action[i] << " ";
  // cout << endl;
  
  MotorTorqueState tor;
  for(int i = 1; i <= N_MOTOR; i++)
    tor.torque[i] = action[i-1];
  // SetAction(cmd, tor);

  control.JointLimit(cmd);
  control.PowerLimit(cmd, state, 1);

  // for(int i = 0; i < 10; i++) {
  //   udp.GetState(state);
  //   PrintMotorState(state);
  // }

  udp.Send(cmd);
  pre = clock();
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
