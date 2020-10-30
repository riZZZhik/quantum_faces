from quantum import Quantum

if __name__ == '__main__':
    q = Quantum((128, 128))

    generated_images = q.generated_images('faces')
