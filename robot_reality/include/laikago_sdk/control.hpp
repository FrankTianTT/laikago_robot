/************************************************************************
Copyright (c) 2018-2019, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef _LAIKAGO_CONTROL_HPP_
#define _LAIKAGO_CONTROL_HPP_

#include "comm.hpp"
#include "loop.hpp"

namespace laikago 
{

	class Control{
	public:
		LOOP loop;
		Control(uint8_t level);
		~Control();
		void InitCmdData(HighCmd& cmd);
		void InitCmdData(LowCmd& cmd);
		void JointLimit(LowCmd&);     	            // only effect under Low Level control in Position mode
		void PowerLimit(LowCmd&, LowState&, int);   /* only effect under Low Level control, input factor: 1~10, 
											        means 10%~100% power limit. If you are new, then use 1; if you are familiar, 
											        then can try bigger number or even comment this function. */

	};

}

#endif
