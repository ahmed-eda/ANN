import mUtils as u

# start the main function
def main():
    print("Starting main call")
    p = u.MyParam()
    p = u.init_param(p)
    p = u.get_data_in_Param(p)
    print("p.y : ", p.y)

    # Then we will start at normalizing data
    print("The End")


if __name__ == "__main__":
    main()
