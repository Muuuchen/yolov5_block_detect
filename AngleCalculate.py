
def adaptx(x): return max(min(x,639),0)

def adapty(y): return max(min(y,479),0)

obj = myCamera()

def calculate_normal_vector(imgdep, mx, my, num = 10):
    '''
    always be matrix (column vector) type in calculations and returns, but fetch by array
    Ax + By + C = z

    :param imgdep: millimeter
    :param mx: pixel
    :param my: pixel
    :param num:
    :return: by angle, by angle, meter, meter(3, )
    '''
    mx = int(mx)
    my = int(my)

    A = np.mat(np.zeros((4 * num, 3)))
    b = np.mat(np.zeros((4 * num, 1)))
    point_all = np.mat(np.zeros((4 * num, 3)))
    point = np.mat(np.zeros((3, 1)))  # XYZ

    counter = 0
    for i in range(-num, num, 1):
        x = adaptx(mx + i)
        y = adapty(my)
        temp = obj.specify(x, y, imgdep[y][x] * 1e-3) # meter

        if temp[0, 0] == 0 or temp[1, 0] == 0 or temp[2, 0] == 0: continue
        A[counter] = [temp[0, 0], temp[1, 0], temp[2, 0]]
        b[counter] = [1]
        point_all[counter] = np.matrix(temp.T[0])
        point = point + np.matrix(temp)
        counter += 1

        x = adaptx(mx)
        y = adapty(my + i)
        temp = obj.specify(x, y, imgdep[y][x] * 1e-3)  # meter

        if temp[0, 0] == 0 or temp[1, 0] == 0 or temp[2, 0] == 0: continue
        A[counter] = [temp[0, 0], temp[1, 0], temp[2, 0]]
        b[counter] = [1]
        point_all[counter] = np.matrix(temp.T[0])
        point = point + np.matrix(temp)
        counter += 1

    point = point / max(counter, 1)
    point = np.array(point.T)[0]

    try:
        X = np.linalg.inv(A.T * A) * A.T * b # X = [[A] [B] [C]]
        horres = math.atan(- X[0, 0] / X[2, 0]) / math.pi * 180
        velres = math.atan(- X[1, 0] / X[2, 0]) / math.pi * 180

        # l0, pix0 = specify(depth_frame, mx, my)
        return horres, velres, np.linalg.norm(point), point
    except:
        traceback.print_exc()
        return math.nan, math.nan, np.linalg.norm(point), point

