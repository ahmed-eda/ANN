import mUtils as u


# start the main function
def main():
    print("Starting main call")

    p = u.MyParam()
    p.our_procedure(_modelNameP='mtest.h5',_loadP=True)
    
    print("The End of main ")


if __name__ == "__main__":
    main()
