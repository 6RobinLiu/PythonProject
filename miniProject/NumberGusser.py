import random

top_range=input("input a number: ")

if top_range.isdigit():
    top_range=int(top_range)

    if top_range<0:
        print("please type a number >0 next time!")
        quit()

else:
    print("please type number next time!")
    quit()



random_number=random.randint(0,top_range)    #任意一个整数  不包括这个数(上界)

guesses=0

while True:
    guesses+=1
    user_guess=input("Make a guess: ")
    if user_guess.isdigit():
        user_guess=int(user_guess)
    else:
        print("please type a number next time.")
        continue
    
    if user_guess==random_number:
        print("right!")
        break
    else:
        if user_guess>random_number:
            print("猜大了🤪")
        else:
            print("猜小了🤞")
                   

print("你猜了",guesses,"次")
    