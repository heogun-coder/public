import numpy as np
from scipy.linalg import lu  # Import LU decomposition from SciPy


def generate_keys(size):
    """Generate keys for the encryption scheme."""
    # Create a random invertible matrix A
    while True:
        A = np.random.randint(1, 10, (size, size))
        if np.linalg.det(A) != 0:  # Ensure A is invertible
            break

    # Perform LU decomposition
    L, U = lu(A)

    # Modify L to create L' (non-invertible)
    L_prime = L.copy()
    L_prime[0, 0] = 0  # Make L' non-invertible

    return (
        L_prime,
        U,
        int(np.round(np.linalg.det(A))),
        A,
    )  # Return public and private keys


def encrypt_message(message, L_prime):
    """Encrypt a numeric message using L'."""
    # Convert message to a 3x3 matrix
    message_matrix = np.array([int(digit) for digit in str(message)]).reshape(3, 3)
    ciphertext = np.dot(message_matrix, L_prime)
    return ciphertext


def decrypt_message(ciphertext, U, det_A):
    """Decrypt a numeric message using U and det(A)."""
    # Compute A' = L'U + det(A) * I
    size = U.shape[0]
    L_prime_U = np.dot(ciphertext, U)
    A_prime = L_prime_U + det_A * np.eye(size)

    # Compute the inverse of A'
    A_prime_inv = np.linalg.inv(A_prime)

    # Recover the original message
    recovered_message = np.dot(L_prime_U, A_prime_inv)
    return np.round(recovered_message).astype(int)


# ========== Main Code Execution ========== #
if __name__ == "__main__":
    # Step 1: Generate keys
    size = 3  # Matrix size for 9-digit numbers
    L_prime, U, det_A, A = generate_keys(size)

    print("L' (Public Key):")
    print(L_prime)
    print("\nU (Private Key):")
    print(U)
    print("\ndet(A):")
    print(det_A)

    # Step 2: Encrypt the message
    original_message = 123234657  # 9-digit numeric message
    print("\nOriginal Message:")
    print(original_message)

    ciphertext = encrypt_message(original_message, L_prime)
    print("\nCiphertext (Encrypted Message):")
    print(ciphertext)

    # Step 3: Decrypt the message
    decrypted_message = decrypt_message(ciphertext, U, det_A)
    print("\nDecrypted Message (Recovered Message):")
    print(decrypted_message)
