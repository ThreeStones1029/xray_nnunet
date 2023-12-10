//MyTimer.h// 
#ifndef __MyTimer_H__  
#define __MyTimer_H__  
#include <windows.h>  

class MyTimer
{
private:
    int _freq;
    LARGE_INTEGER _begin;
    LARGE_INTEGER _end;

public:
    long costTime;      // ���ѵ�ʱ��(��ȷ��΢��)  

public:
    MyTimer()
    {
        LARGE_INTEGER tmp;
        QueryPerformanceFrequency(&tmp);//QueryPerformanceFrequency()���ã�����Ӳ��֧�ֵĸ߾��ȼ�������Ƶ�ʡ�  

        _freq = tmp.QuadPart;
        costTime = 0;
    }

    void Start()      // ��ʼ��ʱ  
    {
        QueryPerformanceCounter(&_begin);//��ó�ʼֵ  
    }

    void End()        // ������ʱ  
    {
        QueryPerformanceCounter(&_end);//�����ֵֹ  
        costTime = (long)((_end.QuadPart - _begin.QuadPart) * 1000000 / _freq);
    }

    void Reset()      // ��ʱ��0  
    {
        costTime = 0;
    }
};
#endif  