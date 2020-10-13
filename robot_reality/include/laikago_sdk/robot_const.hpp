/************************************************************************
Copyright (c) 2018-2019, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef __LAIKAGO_ROBOT_CONST_HPP__
#define __LAIKAGO_ROBOT_CONST_HPP__

namespace laikago 
{
// coordinates
constexpr int X_ = 0;
constexpr int Y_ = 1;
constexpr int Z_ = 2;

// Notice: All of the matrixs equal to two-dimensional arrays,
//         All of the vectors equal to two-dimensional array (column vector) 
//#define MatrixE             {{1,0,0},{0,1,0},{0,0,1}}   // identity matrix
//#define VectorZero          {0,0,0}                     // zero vector
constexpr double MYWidth_Hips   = (0.175f);                    // leg interval (unit: m)
constexpr double MYLength_Thigh = (0.25f);                     // thigh length
constexpr double MYLength_Calf  = (0.25f);                     // calf length
constexpr double MYBodyLength   = (0.432f);                    // length of front legs and rear legs
constexpr double MYLegOffset    = (0.0345f);                   // distance of thigh offset from hip axis
constexpr double MYRockerLength = (0.03f);                     // rocker arm length

// definition of each leg and joint
constexpr int FR_ = 0;       // leg index
constexpr int FL_ = 1;
constexpr int RR_ = 2;
constexpr int RL_ = 3;

constexpr int FR_0 = 1;      // joint index
constexpr int FR_1 = 2;      
constexpr int FR_2 = 3;

constexpr int FL_0 = 4;
constexpr int FL_1 = 5;
constexpr int FL_2 = 6;

constexpr int RR_0 = 7;
constexpr int RR_1 = 8;
constexpr int RR_2 = 9;

constexpr int RL_0 = 10;
constexpr int RL_1 = 11;
constexpr int RL_2 = 12;

constexpr double Hip_max   = 1.047;    // unit:radian ( = 60   degree)
constexpr double Hip_min   = -0.873;   // unit:radian ( = -50  degree)
constexpr double Thigh_max = 3.927;    // unit:radian ( = 225  degree)
constexpr double Thigh_min = -0.524;   // unit:radian ( = -30  degree)
constexpr double Calf_max  = -0.611;   // unit:radian ( = -35  degree)
constexpr double Calf_min  = -2.775;   // unit:radian ( = -159 degree)

}

#endif
