
from offset import KWS

def main(num_k, idx=[]):
    model=KWS(num_k, idx)
    model.forward()
if __name__ == '__main__':
    main(20, [i for i in range(139)])


