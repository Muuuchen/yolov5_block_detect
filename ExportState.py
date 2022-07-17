def exports(sp):
    fb = -1
    sl = -1
    size = sp.blocksize
    if sp.isside != 0 and sp.up_isside == 1:  # or 1 special operate
        if sp.up_square > 20000:  # 5
            size = 5
            if sp.isfivestand:
                print("5 back stand")
                fb = 1
                sl =0
            else:
                print("5 back lie")
                fb = 1
                sl = 1

        else:  # 1
            size = 1
            if sp.isonestand == 0:
                print("1 front stand")
                fb = 0
                sl =0
            else:
                print("1 front lie")
                fb = 0
                sl =1
    elif sp.F_B_pos == 1:
        if sp.velres > 0:
            print("front stand")
            fb = 0
            sl =0
        else:
            print("front lie")
            fb = 0
            sl =1
    elif sp.F_B_pos == 0:
        if sp.velres > 0:
            print("back stand")
            fb = 1
            sl =0
        else:
            print("back lie")
            fb = 1
            sl =1
    elif sp.F_B_pos == 2:
        if sp.up_square > 20000:  # 5
            if sp.isfivefront:
                print("5 back stand")
                fb = 1
                sl =0
            else:
                print("5 back lie")
                fb = 1
                sl =1
        else:  # 1
            if sp.isonefront == 1:
                print("1 front stand")
                fb = 0
                sl =0
            else:
                print("1 front lie")
                fb = 0
                sl =1
    else:
        print("不能判断")
    return fb,sl,size