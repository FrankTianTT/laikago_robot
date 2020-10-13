//
// Created by Fanming Luo on 2019-11-01.
//

#ifndef LAIKAGO_SDK_PARSER_H
#define LAIKAGO_SDK_PARSER_H

#include <fstream>
#include <sstream>
#include "comm.hpp"
namespace laikago
{
void PrintState(const LowState &state, std::ofstream &_of) {
    _of << state.imu.acceleration[0]    << " " << state.imu.acceleration[1]     << " " << state.imu.acceleration[2]     << " "
        << state.imu.gyroscope[0]       << " " << state.imu.gyroscope[1]        << " " << state.imu.gyroscope[2]        << " "
        << state.imu.rpy[0]             << " " << state.imu.rpy[1]              << " " << state.imu.rpy[2]              << " "
        << state.imu.temp               << " "
        << state.imu.quaternion[0]      << " " << state.imu.quaternion[1]       << " " << state.imu.quaternion[2]       << " "
        << state.imu.quaternion[3]      << " "
        << state.footForce[0]           << " " << state.footForce[1]            << " " << state.footForce[2]            << " "
        << state.footForce[3]           << " " << state.tick                    << " " << (int)state.levelFlag          << " ";
    for (int i = 1; i <= 12; ++i) {
        _of << state.motorState[i].position << " " << state.motorState[i].torque << " " << state.motorState[i].velocity << " "
            << state.motorState[i].temperature << " " << (int) state.motorState[i].mode << " ";
    }
    for (int i = 0; i < 40; ++i) {
        _of << (int)state.wirelessRemote[i] << " ";
    }
    _of << std::endl;
}

void PrintState(const HighState &state, std::ofstream &_of) {
    _of << state.imu.acceleration[0]    << " " << state.imu.acceleration[1]     << " " << state.imu.acceleration[2]     << " "
        << state.imu.gyroscope[0]       << " " << state.imu.gyroscope[1]        << " " << state.imu.gyroscope[2]        << " "
        << state.imu.rpy[0]             << " " << state.imu.rpy[1]              << " " << state.imu.rpy[2]              << " "
        << state.imu.temp               << " "
        << state.imu.quaternion[0]      << " " << state.imu.quaternion[1]       << " " << state.imu.quaternion[2]       << " "
        << state.imu.quaternion[3]      << " "
        << state.footForce[0]           << " " << state.footForce[1]            << " " << state.footForce[2]            << " "
        << state.footForce[3]           << " " << state.tick                    << " " << (int)state.levelFlag          << " "
        << state.bodyHeight             << " " << (int)state.mode               << " "
        << state.footPosition2Body[0].x << " " << state.footPosition2Body[0].y  << " " << state.footPosition2Body[0].z  << " "
        << state.footPosition2Body[1].x << " " << state.footPosition2Body[1].y  << " " << state.footPosition2Body[1].z  << " "
        << state.footPosition2Body[2].x << " " << state.footPosition2Body[2].y  << " " << state.footPosition2Body[2].z  << " "
        << state.footPosition2Body[3].x << " " << state.footPosition2Body[3].y  << " " << state.footPosition2Body[3].z  << " "
        << state.footSpeed2Body[0].x    << " " << state.footSpeed2Body[0].y     << " " << state.footSpeed2Body[0].z     << " "
        << state.footSpeed2Body[1].x    << " " << state.footSpeed2Body[1].y     << " " << state.footSpeed2Body[1].z     << " "
        << state.footSpeed2Body[2].x    << " " << state.footSpeed2Body[2].y     << " " << state.footSpeed2Body[2].z     << " "
        << state.footSpeed2Body[3].x    << " " << state.footSpeed2Body[3].y     << " " << state.footSpeed2Body[3].z     << " "
        << state.forwardSpeed           << " " << state.forwardSpeed            << " " << state.rotateSpeed             << " "
        << state.sideSpeed              << " " << state.updownSpeed             << " "
        << state.forwardPosition.x      << " " << state.forwardPosition.y       << " " << state.forwardPosition.z       << " "
        << state.sidePosition.x         << " " << state.sidePosition.y          << " " << state.sidePosition.z          << " ";

    for (int i = 0; i < 40; ++i) {
        _of << (int)state.wirelessRemote[i] << " ";
    }
    _of << std::endl;
}

void ReadState(LowState &state, std::ifstream &_if) {
    std::stringstream ss;
    ss.clear();
    std::string _str;
    std::getline(_if, _str);
    ss.str(_str);
     ss >> state.imu.acceleration[0]   >> state.imu.acceleration[1]   >> state.imu.acceleration[2]
        >> state.imu.gyroscope[0]      >> state.imu.gyroscope[1]      >> state.imu.gyroscope[2]
        >> state.imu.rpy[0]            >> state.imu.rpy[1]            >> state.imu.rpy[2]
        >> state.imu.temp
        >> state.imu.quaternion[0]     >> state.imu.quaternion[1]     >> state.imu.quaternion[2]
        >> state.imu.quaternion[3]
        >> state.footForce[0]          >> state.footForce[1]          >> state.footForce[2]
        >> state.footForce[3]          >> state.tick                  >> state.levelFlag;

     for (int i = 1; i <= 12; ++i) {
        ss >> state.motorState[i].position >> state.motorState[i].torque >> state.motorState[i].velocity
            >> state.motorState[i].temperature >> state.motorState[i].mode;
     }

     for (int i = 0; i < 40; ++i) {
         ss >> state.wirelessRemote[i];
     }
}

void ReadState(HighState &state, std::ifstream &_if) {
    std::stringstream ss;
    ss.clear();
    std::string _str;
    std::getline(_if, _str);
    ss.str(_str);   
     ss >> state.imu.acceleration[0]     >> state.imu.acceleration[1]      >> state.imu.acceleration[2]
        >> state.imu.gyroscope[0]        >> state.imu.gyroscope[1]         >> state.imu.gyroscope[2]
        >> state.imu.rpy[0]              >> state.imu.rpy[1]               >> state.imu.rpy[2]
        >> state.imu.temp
        >> state.imu.quaternion[0]       >> state.imu.quaternion[1]        >> state.imu.quaternion[2]
        >> state.imu.quaternion[3]
        >> state.footForce[0]            >> state.footForce[1]             >> state.footForce[2]
        >> state.footForce[3]            >> state.tick                     >> state.levelFlag
        >> state.bodyHeight              >> state.mode
        >> state.footPosition2Body[0].x  >> state.footPosition2Body[0].y   >> state.footPosition2Body[0].z
        >> state.footPosition2Body[1].x  >> state.footPosition2Body[1].y   >> state.footPosition2Body[1].z
        >> state.footPosition2Body[2].x  >> state.footPosition2Body[2].y   >> state.footPosition2Body[2].z
        >> state.footPosition2Body[3].x  >> state.footPosition2Body[3].y   >> state.footPosition2Body[3].z
        >> state.footSpeed2Body[0].x     >> state.footSpeed2Body[0].y      >> state.footSpeed2Body[0].z
        >> state.footSpeed2Body[1].x     >> state.footSpeed2Body[1].y      >> state.footSpeed2Body[1].z
        >> state.footSpeed2Body[2].x     >> state.footSpeed2Body[2].y      >> state.footSpeed2Body[2].z
        >> state.footSpeed2Body[3].x     >> state.footSpeed2Body[3].y      >> state.footSpeed2Body[3].z
        >> state.forwardSpeed            >> state.forwardSpeed             >> state.rotateSpeed
        >> state.sideSpeed               >> state.updownSpeed
        >> state.forwardPosition.x       >> state.forwardPosition.y        >> state.forwardPosition.z
        >> state.sidePosition.x          >> state.sidePosition.y           >> state.sidePosition.z          ;

    for (int i = 0; i < 40; ++i) {
        ss >> state.wirelessRemote[i] ;
    }
}

void PrintCmd(const LowCmd &cmd, std::ofstream &_of) {
    _of << (int)cmd.levelFlag       << " " ;
    for (int i = 1 ; i <= 12; ++i) {
        _of << (int)cmd.motorCmd[i].mode     << " " << cmd.motorCmd[i].velocity << " " << cmd.motorCmd[i].position << " "
            << cmd.motorCmd[i].torque   << " " << cmd.motorCmd[i].positionStiffness << " " << cmd.motorCmd[i].velocityStiffness << " ";
    }
    for (int i = 0;i < 40; ++i) {
        _of << (int)cmd.wirelessRemote[i] << " ";
    }
    _of << std::endl;
}

void PrintCmd(const HighCmd &cmd, std::ofstream &_of) {
    _of << (int) cmd.levelFlag << " " << (int)cmd.mode << " " << cmd.sideSpeed << " " << cmd.rotateSpeed << " " 
        << cmd.forwardSpeed << " " << cmd.bodyHeight << " " << cmd.pitch << " " << cmd.roll << " " << cmd.yaw << " "
        << cmd.footRaiseHeight << " ";
    for (int i = 0; i < 40; ++i) {
        _of << (int) cmd.wirelessRemote[i] << " ";
    }
    _of << std::endl;
}

void ReadCmd(HighCmd &cmd, std::ifstream &_if) {
    std::stringstream ss;
    ss.clear();
    std::string _str;
    std::getline(_if, _str);
    ss.str(_str);
    ss >> cmd.levelFlag  >> cmd.mode  >> cmd.sideSpeed  >> cmd.rotateSpeed 
        >> cmd.forwardSpeed  >> cmd.bodyHeight  >> cmd.pitch  >> cmd.roll  >> cmd.yaw 
        >> cmd.footRaiseHeight ;
    for (int i = 0; i < 40; ++i) {
        ss >> cmd.wirelessRemote[i] ;
    } 
}

void ReadCmd(LowCmd &cmd, std::ifstream &_if) {
    std::stringstream ss;
    ss.clear();
    std::string _str;
    std::getline(_if, _str);
    ss.str(_str);
    ss >> cmd.levelFlag        ;
    for (int i = 1 ; i <= 12; ++i) {
        ss >> cmd.motorCmd[i].mode      >> cmd.motorCmd[i].velocity  >> cmd.motorCmd[i].position
            >> cmd.motorCmd[i].torque    >> cmd.motorCmd[i].positionStiffness  >> cmd.motorCmd[i].velocityStiffness ;
    }
    for (int i = 0;i < 40; ++i) {
        ss >> cmd.wirelessRemote[i] ;
    }
}
}// namespace
#endif //LAIKAGO_SDK_PARSER_H
