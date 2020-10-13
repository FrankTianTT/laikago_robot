/************************************************************************
Copyright (c) 2018-2019, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef _LAIKAGO_UDP_HPP_
#define _LAIKAGO_UDP_HPP_

#include "comm.hpp"
#include <pthread.h>

namespace laikago
{

    constexpr int UDP_CLIENT_PORT = 8008;                // local port
    constexpr int UDP_SERVER_PORT = 8007;                // target port
    constexpr char UDP_SERVER_IP[] = "192.168.123.10";   // target IP address

    // Notice: User defined data(like struct) should add crc(4Byte) at the end.
    class UDP {
	public:
        UDP(int sendLength, int recvLength);
        UDP(uint16_t localPort, const char* targetIP, uint16_t targetPort, int sendLength, int recvLength);
        ~UDP();
		int Send(HighCmd&);
        int Send(LowCmd&);
        int Send(char* cmd);
        int Recv(); // directly save in buffer
        void GetState(HighState&);
        void GetState(LowState&);
        void GetState(char* recv);
        UDPState State;
    private:
        int sendLength;
        int recvLength;
        char* recvBuf;
        char* recvSource;
        pthread_mutex_t stateMut;
        pthread_mutex_t socketMut;
        int mySend(char* cmd);
        int myRecv();
	};

}

#endif
