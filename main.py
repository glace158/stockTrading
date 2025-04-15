import sys

from richdog import RichDogTrain, RichDogTest

if __name__ == '__main__':
    
    arg = sys.argv

    if len(arg)<= 1:        
        arg = ["main.py", "train", "PPO_preTrained/Richdog/PPO_Richdog_0_20250403-141334.pth"]
        

    if len(arg) > 1:
        if arg[1] == 'train':
            richdog = RichDogTrain()
            richdog.train()
        elif arg[1] == 'test':
            richdog = RichDogTest(checkpoint_path=arg[2])
            richdog.test()
