def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(start_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
