#include "frame_grabber.h"

#include <opencv2/core/mat.hpp>

template <typename T>
FrameGrabber<T>::FrameGrabber(size_t max_size) : max_size(max_size) {}

template <typename T>
void FrameGrabber<T>::push(const T& item) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]() { return queue.size() < max_size; });
    queue.push(item);
    cv.notify_all();
}

template <typename T>
bool FrameGrabber<T>::pop(T& item) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]() { return !queue.empty(); });
    item = queue.front();
    queue.pop();
    cv.notify_all();
    return true;
}

template <typename T>
bool FrameGrabber<T>::empty() {
    std::lock_guard<std::mutex> lock(mtx);
    return queue.empty();
}

template class FrameGrabber<cv::UMat>;