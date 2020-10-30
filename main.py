from quantum import Quantum

if __name__ == '__main__':
    q = Quantum((32, 32))

    generated_images = q.generate_images('faces')
