#!/bin/python3

#########################################################################################
# Date: 2023-08-03
# Creator: Pascal Müller
# Version:0.1
#########################################################################################

import multiprocessing

#-----------------------------------------------------------------------------------------
# CPU_Multi_Tasker
#-----------------------------------------------------------------------------------------

class CPUMultiTasker:

    
    #---------------CONSTRUCTOR---------------
    def __init__(self):

        self.num_cores = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(processes=self.num_cores)
        self.queue = multiprocessing.Queue()
    #-----------------------------------------     

    def _worker(self, func, args, kwargs):

        result = func(*args, **kwargs)
        self.queue.put(result)

    #-----------------------

    def run_function(self, func, args=(), kwargs={}):

        # If all cores are busy, add the task to the queue
        if self.pool._taskqueue.qsize() >= self.num_cores:
            print("All cores busy, adding task to the queue.")
            self.queue.put(None)  # Placeholder for the queue
            self.queue.put((func, args, kwargs))

        else:
            # If a core is available, run the task immediately
            self.pool.apply_async(self._worker, args=(func, args, kwargs))

    #-----------------------

    def close(self):
        
        self.pool.close()
        self.pool.join()