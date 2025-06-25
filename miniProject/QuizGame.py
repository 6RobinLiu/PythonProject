print("Welcome to Quiz Game!")


playing=input("do you want to play?(yes/no)   ")

if playing.lower() !="yes":     #lower函数把所有的 内容小写  YEs YES 等等  都变成小写 
    quit()
    
print("ok,let's play! 😀")

### first problem
answer=input("What does CPU stand for?  ")

if answer=="central processing unit":
    print("✔")
else:
    print("不对哦✋")
    
#### other problems
###   变量可以直接用
