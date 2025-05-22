import sys

from richdog import RichDogTrain, RichDogTest

if __name__ == '__main__':
    
    arg = sys.argv

    if len(arg)<= 1:        
        #arg = ["main.py", "train", "PPO_preTrained/MountainCarContinuous-v0/PPO_MountainCarContinuous-v0_0_20250514-224951.pth"]
        arg = ["main.py", "train", ""]
        

    if len(arg) > 1:
        if arg[1] == 'train':
            richdog = RichDogTrain()
            richdog.random_train()
        elif arg[1] == 'test':
            richdog = RichDogTest(checkpoint_path=arg[2])
            richdog.test()
