#ifndef BARRIER_H
#define BARRIER_H

#include <QMutex>
#include <QWaitCondition>
#include <QSharedPointer>

// Data "pimpl" class (not to be used directly)
class BarrierData
{
public:
    BarrierData(int count, int NumCam) : count(count), NumCam(NumCam) {}

    void wait() {
        mutex.lock();
        --count;
        if (count > 0){
            condition.wait(&mutex);
        }
        else{
            condition.wakeAll();
            // When all the threads are awake the count is restarted
            count = NumCam;
        }
        mutex.unlock();
    }
private:
    Q_DISABLE_COPY(BarrierData)
    int count;
    int NumCam;
    QMutex mutex;
    QWaitCondition condition;
};

class Barrier {
public:
    // Create a barrier that will wait for count threads
    Barrier(int count, int NumCam) : d(new BarrierData(count, NumCam)) {}
    void wait() {
        d->wait();
    }

private:
    QSharedPointer<BarrierData> d;
};

#endif // BARRIER_H
