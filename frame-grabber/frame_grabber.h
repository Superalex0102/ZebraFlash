#ifndef FRAME_GRABBER_H
#define FRAME_GRABBER_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class FrameGrabber {
public:
    FrameGrabber(size_t max_size = 10);
    void push(const T& item);
    bool pop(T& item);
    bool empty();

private:
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
    const size_t max_size;
};

#endif // FRAME_GRABBER_H