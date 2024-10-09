

def fitness(positions, inputnum, hiddennum, outputnum, net, inputn, outputn, output_train, inputn_test, outputs, output_test):
    # 取得权值与偏置
    w1 = positions[0:inputnum*hiddennum]
    b1 = positions[inputnum*hiddennum : inputnum*hiddennum+hiddennum]
    w2 = positions[inputnum*hiddennum+hiddennum : inputnum*hiddennum+hiddennum+hiddennum*outputnum]
    b2 = positions[inputnum*hiddennum+hiddennum+hiddennum*outputnum : inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum]
    
    # 网络权值赋值
    net.