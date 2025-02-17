import tensorflow as tf
import os

def check_gpu_status():
    print("TensorFlow version:", tf.__version__)

    # GPU 확인
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. TensorFlow will run on CPU.")
        return

    print(f"Number of GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.name}")

    # GPU 메모리 동적 할당 설정 확인
    for gpu in gpus:
        memory_growth = tf.config.experimental.get_memory_growth(gpu)
        print(f"Memory growth enabled for {gpu.name}: {memory_growth}")

    # GPU로 간단한 연산 테스트
    try:
        with tf.device('/GPU:0'):  # GPU가 0번으로 인식되는지 확인
            print("Running test computation on GPU...")
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = a + b
            print("Computation result:", c.numpy())
    except RuntimeError as e:
        print("Error during GPU test computation:", e)

if __name__ == "__main__":
    check_gpu_status()
