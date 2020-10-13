/************************************************************************
Copyright (c) 2018-2019, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef _LAIKAGO_LOOP_HPP_
#define _LAIKAGO_LOOP_HPP_

#include "comm.hpp"
#include <pthread.h>

namespace laikago 
{

	constexpr int PRIORITY_CMD    = 99;   // real-time priority
	constexpr int PRIORITY_STATE  = 99;
	constexpr int CPU_UDP   = 4;          // cpu affinity
	constexpr int CPU_LCM   = 5;          // here use the same one because of performance restriction
	constexpr int CPU_PRINT = 0;          // cpu 0 

	typedef void(*PointerToFunction)();

	class LOOP{
	public:
		LOOP();
		~LOOP();
		void SetLCM(bool);
		void SetPrint(bool);
        void SetUDPPeriod(int);       // 2000~20000(us) => 50~500(Hz). Do not use while controlling real robot.
		void SetLCMPeriod(int);
		void SetPrintPeriod(int); 
		int Start(void);              // fall into loop and will not return
		void RegistFunc(const char* topic, PointerToFunction pf); // regist callback function. topic is the entrance to three different loop.
		                                                          // topic = "UDP/Send", "UDP/Recv", "LCM/Recv", "PRINT".

	};

}

#endif
