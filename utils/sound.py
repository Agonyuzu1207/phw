import winsound

class sound:
    def play(self):
        frequency=2500
        duration=1000
        winsound.Beep(frequency,duration)