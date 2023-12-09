import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sympy
import random
from math import log, pi, cos, sin, sqrt
import tempfile
import os, sys
import Pyfhel
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

# PyCrCNN Source: https://github.com/AlexMV12/PyCrCNN
# Pyfhel Source: https://github.com/ibarrond/Pyfhel

class SecretKey:

    """An instance of a secret key.

    The secret key consists of one polynomials generated
    from key_generator.py.

    Attributes:
        s (Polynomial): Secret key.
    """

    def __init__(self, s):
        """Sets public key to given inputs.

        Args:
            s (Polynomial): Secret key.
        """
        self.s = s

    def __str__(self):
        """Represents secret key as a string.

        Returns:
            A string which represents the secret key.
        """
        return str(self.s)

class RotationKey:

    """An instance of a rotation key.

    The rotation key consists of a value determined by the rotation value r.

    Attributes:
        rotation (int): Rotation value r.
        key (PublicKey): Key values.
    """

    def __init__(self, r, key):
        """Sets rotation key to given inputs.

        Args:
            r (int): Value to be rotated by.
            key (PublicKey): Key.
        """
        self.rotation = r
        self.key = key

    def __str__(self):
        """Represents RotationKey as a string.

        Returns:
            A string which represents the RotationKey.
        """
        return 'Rotation: ' + str(self.rotation) + '\nr0: ' + str(self.key.p0) + '\nr1: ' + str(self.key.p1)


class PublicKey:

    """An instance of a public key.

    The public key consists of a pair of polynomials generated
    from key_generator.py.

    Attributes:
        p0 (Polynomial): First element of public key.
        p1 (Polynomial): Second element of public key.
    """

    def __init__(self, p0, p1):
        """Sets public key to given inputs.

        Args:
            p0 (Polynomial): First element of public key.
            p1 (Polynomial): Second element of public key.
        """
        self.p0 = p0
        self.p1 = p1

    def __str__(self):
        """Represents PublicKey as a string.

        Returns:
            A string which represents the PublicKey.
        """
        return 'p0: ' + str(self.p0) + '\n + p1: ' + str(self.p1)

def mod_exp(val, exp, modulus):
    """Computes an exponent in a modulus.

    Raises val to power exp in the modulus without overflowing.

    Args:
        val (int): Value we wish to raise the power of.
        exp (int): Exponent.
        modulus (int): Modulus where computation is performed.

    Returns:
        A value raised to a power in a modulus.
    """
    return pow(int(val), int(exp), int(modulus))

def mod_inv(val, modulus):
    """Finds an inverse in a given prime modulus.

    Finds the inverse of val in the modulus.

    Args:
        val (int): Value to find the inverse of.
        modulus (int): Modulus where computation is performed.
            Note: MUST BE PRIME.

    Returns:
        The inverse of the given value in the modulus.
    """
    return mod_exp(val, modulus - 2, modulus)

def find_generator(modulus):
    """Finds a generator in the given modulus.

    Finds a generator, or primitive root, in the given prime modulus.

    Args:
        modulus (int): Modulus to find the generator in. Note: MUST
            BE PRIME.

    Returns:
        A generator, or primitive root in the given modulus.
    """
    return sympy.ntheory.primitive_root(modulus)

def root_of_unity(order, modulus):
    """Finds a root of unity in the given modulus.

    Finds a root of unity with the given order in the given prime modulus.

    Args:
        order (int): Order n of the root of unity (an nth root of unity).
        modulus (int): Modulus to find the root of unity in. Note: MUST BE
            PRIME

    Returns:
        A root of unity with the given order in the given modulus.
    """
    # if ((modulus - 1) % order) != 0:
    #     raise ValueError('Must have order q | m - 1, where m is the modulus. \
    #         The values m = ' + str(modulus) + ' and q = ' + str(order) + ' do not satisfy this.')

    generator = find_generator(modulus)
    # if generator is None:
    #     raise ValueError('No primitive root of unity mod m = ' + str(modulus))

    result = mod_exp(generator, (modulus - 1)//order, modulus)

    if result == 1:
        return root_of_unity(order, modulus)

    return result

def is_prime(number, num_trials=200):
    """Determines whether a number is prime.

    Runs the Miller-Rabin probabilistic primality test many times on the given number.

    Args:
        number (int): Number to perform primality test on.
        num_trials (int): Number of times to perform the Miller-Rabin test.

    Returns:
        True if number is prime, False otherwise.
    """
    if number < 2:
        return False
    if number != 2 and number % 2 == 0:
        return False

    # Find largest odd factor of n-1.
    exp = number - 1
    while exp % 2 == 0:
        exp //= 2

    for _ in range(num_trials):
        rand_val = int(random.SystemRandom().randrange(1, number))
        new_exp = exp
        power = pow(rand_val, new_exp, number)
        while new_exp != number - 1 and power != 1 and power != number - 1:
            power = (power * power) % number
            new_exp *= 2
        if power != number - 1 and new_exp % 2 == 0:
            return False

    return True

class CRTContext:

    """An instance of Chinese Remainder Theorem parameters.

    We split a large number into its prime factors.

    Attributes:
        poly_degree (int): Polynomial ring degree.
        primes (list): List of primes.
        modulus (int): Large modulus, product of all primes.
    """

    def __init__(self, num_primes, prime_size, poly_degree):
        """Inits CRTContext with a list of primes.

        Args:
            num_primes (int): Number of primes.
            prime_size (int): Minimum number of bits in primes.
            poly_degree (int): Polynomial degree of ring.
        """
        self.poly_degree = poly_degree
        self.generate_primes(num_primes, prime_size, mod=2*poly_degree)
        self.generate_ntt_contexts()

        self.modulus = 1
        for prime in self.primes:
            self.modulus *= prime

        self.precompute_crt()

    def generate_primes(self, num_primes, prime_size, mod):
        """Generates primes that are 1 (mod M), where M is twice the polynomial degree.

        Args:
            num_primes (int): Number of primes.
            prime_size (int): Minimum number of bits in primes.
            mod (int): Value M (must be a power of two) such that primes are 1 (mod M).
        """
        self.primes = [1] * num_primes
        possible_prime = (1 << prime_size) + 1
        for i in range(num_primes):
            possible_prime += mod
            while not is_prime(possible_prime):
                possible_prime += mod
            self.primes[i] = possible_prime

    def generate_ntt_contexts(self):
        """Generates NTTContexts for each primes.
        """
        self.ntts = []
        for prime in self.primes:
            ntt = NTTContext(self.poly_degree, prime, root_of_unity)
            self.ntts.append(ntt)

    def precompute_crt(self):
        """Perform precomputations required for switching representations.
        """
        num_primes = len(self.primes)
        self.crt_vals = [1] * num_primes
        self.crt_inv_vals = [1] * num_primes
        for i in range(num_primes):
            self.crt_vals[i] = self.modulus // self.primes[i]
            self.crt_inv_vals[i] = mod_inv(self.crt_vals[i], self.primes[i])

    def crt(self, value):
        """Transform value to CRT representation.

        Args:
            value (int): Value to be transformed to CRT representation.
            primes (list): List of primes to use for CRT representation.
        """
        return [value % p for p in self.primes]

    def reconstruct(self, values):
        """Reconstructs original value from vals from the CRT representation to the regular representation.

        Args:
            values (list): List of values which are x_i (mod p_i).
        """
        assert len(values) == len(self.primes)
        regular_rep_val = 0

        for i in range(len(values)):
            intermed_val = (values[i] * self.crt_inv_vals[i]) % self.primes[i]
            intermed_val = (intermed_val * self.crt_vals[i]) % self.modulus
            regular_rep_val += intermed_val
            regular_rep_val %= self.modulus

        return regular_rep_val

class NTTContext:
    """An instance of Number/Fermat Theoretic Transform parameters.

    Here, R is the quotient ring Z_a[x]/f(x), where f(x) = x^d + 1.
    The NTTContext keeps track of the ring degree d, the coefficient
    modulus a, a root of unity w so that w^2d = 1 (mod a), and
    precomputations to perform the NTT/FTT and the inverse NTT/FTT.

    Attributes:
        coeff_modulus (int): Modulus for coefficients of the polynomial.
        degree (int): Degree of the polynomial ring.
        roots_of_unity (list): The ith member of the list is w^i, where w
            is a root of unity.
        roots_of_unity_inv (list): The ith member of the list is w^(-i),
            where w is a root of unity.
        scaled_rou_inv (list): The ith member of the list is 1/n * w^(-i),
            where w is a root of unity.
        reversed_bits (list): The ith member of the list is the bits of i
            reversed, used in the iterative implementation of NTT.
    """

    def __init__(self, poly_degree, coeff_modulus, root_of_unity=None):
        """Inits NTTContext with a coefficient modulus for the polynomial ring
        Z[x]/f(x) where f has the given poly_degree.

        Args:
            poly_degree (int): Degree of the polynomial ring.
            coeff_modulus (int): Modulus for coefficients of the polynomial.
            root_of_unity (int): Root of unity to perform the NTT with. If it
                takes its default value of None, we compute a root of unity to
                use.
        """
        assert (poly_degree & (poly_degree - 1)) == 0, \
            "Polynomial degree must be a power of 2. d = " + str(poly_degree) + " is not."
        self.coeff_modulus = coeff_modulus
        self.degree = poly_degree
        # We use the (2d)-th root of unity, since d of these are roots of x^d + 1, which can be
        # used to uniquely identify any polynomial mod x^d + 1 from the CRT representation of
        # x^d + 1.
        root_of_unity = root_of_unity(order=2 * poly_degree, modulus=coeff_modulus)

        self.precompute_ntt(root_of_unity)

    def precompute_ntt(self, root_of_unity):
        """Performs precomputations for the NTT and inverse NTT.

        Precomputes all powers of roots of unity for the NTT and scaled powers of inverse
        roots of unity for the inverse NTT.

        Args:
            root_of_unity (int): Root of unity to perform the NTT with.
        """

        # Find powers of root of unity.
        self.roots_of_unity = [1] * self.degree
        for i in range(1, self.degree):
            self.roots_of_unity[i] = \
                (self.roots_of_unity[i - 1] * root_of_unity) % self.coeff_modulus

        # Find powers of inverse root of unity.
        root_of_unity_inv = mod_inv(root_of_unity, self.coeff_modulus)
        self.roots_of_unity_inv = [1] * self.degree
        for i in range(1, self.degree):
            self.roots_of_unity_inv[i] = \
                (self.roots_of_unity_inv[i - 1] * root_of_unity_inv) % self.coeff_modulus

        # Compute precomputed array of reversed bits for iterated NTT.
        self.reversed_bits = [0] * self.degree
        width = int(log(self.degree, 2))
        for i in range(self.degree):
            self.reversed_bits[i] = reverse_bits(i, width) % self.degree

    def ntt(self, coeffs, rou):
        """Runs NTT on the given coefficients.

        Runs iterated NTT with the given coefficients and roots of unity. See
        paper for pseudocode.

        Args:
            coeffs (list): List of coefficients to transform. Must be the
                length of the polynomial degree.
            rou (list): Powers of roots of unity to be used for transformation.
                For inverse NTT, this is the powers of the inverse root of unity.

        Returns:
            List of transformed coefficients.
        """
        num_coeffs = len(coeffs)
        assert len(rou) == num_coeffs, \
            "Length of the roots of unity is too small. Length is " + len(rou)

        result = bit_reverse_vec(coeffs)

        log_num_coeffs = int(log(num_coeffs, 2))

        for logm in range(1, log_num_coeffs + 1):
            for j in range(0, num_coeffs, (1 << logm)):
                for i in range(1 << (logm - 1)):
                    index_even = j + i
                    index_odd = j + i + (1 << (logm - 1))

                    rou_idx = (i << (1 + log_num_coeffs - logm))
                    omega_factor = (rou[rou_idx] * result[index_odd]) % self.coeff_modulus

                    butterfly_plus = (result[index_even] + omega_factor) % self.coeff_modulus
                    butterfly_minus = (result[index_even] - omega_factor) % self.coeff_modulus

                    result[index_even] = butterfly_plus
                    result[index_odd] = butterfly_minus

        return result

    def ftt_fwd(self, coeffs):
        """Runs forward FTT on the given coefficients.

        Runs forward FTT with the given coefficients and parameters in the context.

        Args:
            coeffs (list): List of coefficients to transform. Must be the
                length of the polynomial degree.

        Returns:
            List of transformed coefficients.
        """
        num_coeffs = len(coeffs)
        assert num_coeffs == self.degree, "ftt_fwd: input length does not match context degree"

        # We use the FTT input given in the FTT paper.
        ftt_input = [(int(coeffs[i]) * self.roots_of_unity[i]) % self.coeff_modulus
                     for i in range(num_coeffs)]

        return self.ntt(coeffs=ftt_input, rou=self.roots_of_unity)

    def ftt_inv(self, coeffs):
        """Runs inverse FTT on the given coefficients.

        Runs inverse FTT with the given coefficients and parameters in the context.

        Args:
            coeffs (list): List of coefficients to transform. Must be the
                length of the polynomial degree.

        Returns:
            List of inversely transformed coefficients.
        """
        num_coeffs = len(coeffs)
        assert num_coeffs == self.degree, "ntt_inv: input length does not match context degree"

        to_scale_down = self.ntt(coeffs=coeffs, rou=self.roots_of_unity_inv)
        poly_degree_inv = mod_inv(self.degree, self.coeff_modulus)

        # We scale down the FTT output given in the FTT paper.
        result = [(int(to_scale_down[i]) * self.roots_of_unity_inv[i] * poly_degree_inv) \
                  % self.coeff_modulus for i in range(num_coeffs)]

        return result

def sample_uniform(min_val, max_val, num_samples):
    """Samples from a uniform distribution.

    Samples num_samples integer values from the range [min, max)
    uniformly at random.

    Args:
        min_val (int): Minimum value (inclusive).
        max_val (int): Maximum value (exclusive).
        num_samples (int): Number of samples to be drawn.

    Returns:
        A list of randomly sampled values.
    """
    if num_samples == 1:
        #return random.SystemRandom().randrange(min_val, max_val)
        return random.randrange(min_val, max_val)
    #return [random.SystemRandom().randrange(min_val, max_val)
    #    for _ in range(num_samples)]
    return [random.randrange(min_val, max_val)
        for _ in range(num_samples)]

def sample_triangle(num_samples):
    """Samples from a discrete triangle distribution.

    Samples num_samples values from [-1, 0, 1] with probabilities
    [0.25, 0.5, 0.25], respectively.

    Args:
        num_samples (int): Number of samples to be drawn.

    Returns:
        A list of randomly sampled values.
    """
    sample = [0] * num_samples

    for i in range(num_samples):
        # r = random.SystemRandom().randrange(0, 4)
        r = random.randrange(0, 4)
        if r == 0: sample[i] = -1
        elif r == 1: sample[i] = 1
        else: sample[i] = 0
    return sample

def reverse_bits(value, width):
    """Reverses bits of an integer.

    Reverse bits of the given value with a specified bit width.
    For example, reversing the value 6 = 0b110 with a width of 5
    would result in reversing 0b00110, which becomes 0b01100 = 12.

    Args:
        value (int): Value to be reversed.   
        width (int): Number of bits to consider in reversal.

    Returns:
        The reversed int value of the input.
    """
    binary_val = '{:0{width}b}'.format(value, width=width)
    return int(binary_val[::-1], 2)

def bit_reverse_vec(values):
    """Reverses list by reversing the bits of the indices.

    Reverse indices of the given list.
    For example, reversing the list [0, 1, 2, 3, 4, 5, 6, 7] would become
    [0, 4, 2, 6, 1, 5, 3, 7], since 1 = 0b001 reversed is 0b100 = 4,
    3 = 0b011 reversed is 0b110 = 6.

    Args:
        values (list): List of values to be reversed. Length of list must be a power of two. 

    Returns:
        The reversed list based on indices.
    """
    result = [0] * len(values)
    for i in range(len(values)):
        result[i] = values[reverse_bits(i, int(log(len(values), 2)))]
    return result
  
class CKKSParameters:

    """An instance of parameters for the CKKS scheme.

    Attributes:
        poly_degree (int): Degree d of polynomial that determines the
            quotient ring R.
        ciph_modulus (int): Coefficient modulus of ciphertexts.
        big_modulus (int): Large modulus used for bootstrapping.
        scaling_factor (float): Scaling factor to multiply by.
        hamming_weight (int): Hamming weight parameter for sampling secret key.
        taylor_iterations (int): Number of iterations to perform for Taylor series in
            bootstrapping.
        prime_size (int): Minimum number of bits in primes for RNS representation.
        crt_context (CRTContext): Context to manage RNS representation.
    """

    def __init__(self, poly_degree, ciph_modulus, big_modulus, scaling_factor, taylor_iterations=6,
                 prime_size=59):
        """Inits Parameters with the given parameters.

        Args:
            poly_degree (int): Degree d of polynomial of ring R.
            ciph_modulus (int): Coefficient modulus of ciphertexts.
            big_modulus (int): Large modulus used for bootstrapping.
            scaling_factor (float): Scaling factor to multiply by.
            taylor_iterations (int): Number of iterations to perform for Taylor series in
                bootstrapping.
            prime_size (int): Minimum number of bits in primes for RNS representation. Can set to 
                None if using the RNS representation if undesirable.
        """
        self.poly_degree = poly_degree
        self.ciph_modulus = ciph_modulus
        self.big_modulus = big_modulus
        self.scaling_factor = scaling_factor
        self.num_taylor_iterations = taylor_iterations
        self.hamming_weight = poly_degree // 4
        self.crt_context = None

        if prime_size:
            num_primes = 1 + int((1 + log(poly_degree, 2) + 4 * log(big_modulus, 2) \
             / prime_size))
            self.crt_context = CRTContext(num_primes, prime_size, poly_degree)

    def print_parameters(self):
        """Prints parameters.
        """
        print("Encryption parameters")
        print("\t Polynomial degree: %d" %(self.poly_degree))
        print("\t Ciphertext modulus size: %d bits" % (int(log(self.ciph_modulus, 2))))
        print("\t Big ciphertext modulus size: %d bits" % (int(log(self.big_modulus, 2))))
        print("\t Scaling factor size: %d bits" % (int(log(self.scaling_factor, 2))))
        print("\t Number of Taylor iterations: %d" % (self.num_taylor_iterations))
        if self.crt_context:
            rns = "Yes"
        else:
            rns = "No"
        print("\t RNS: %s" % (rns))

############################## PyCrCNN imports ##############################

class HE:
    def generate_keys(self):
        pass

    def generate_relin_keys(self):
        pass

    def get_public_key(self):
        pass

    def get_relin_key(self):
        pass

    def load_public_key(self, key):
        pass

    def load_relin_key(self, key):
        pass

    def encode_matrix(self, matrix):
        """Encode a matrix in a plaintext HE nD-matrix.

        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded

        Returns
        -------
        matrix
            nD-np.array with encoded values
        """
        pass

    def decode_matrix(self, matrix):
        pass

    def encrypt_matrix(self, matrix):
        pass

    def decrypt_matrix(self, matrix):
        pass

    def encode_number(self, number):
        pass

    def power(self, number, exp):
        pass

    def noise_budget(self, ciphertext):
        pass

class CKKSPyfhel(HE):
    def __init__(self, m=16384, scale=2**30, qi=[31, 30, 30, 30, 30, 30, 30, 30, 30, 31]):
        self.he = Pyfhel()
        self.he.contextGen(scheme='ckks', n=m, scale=scale, qi_sizes=qi)

    def encode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.encodeFrac(np.array([x], dtype=np.float64))

    def decode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.decodeFrac(x)[0]

    def encrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.encryptFrac(np.array([x], dtype=np.float64))

    def decrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.decrypt(x)[0]

    def generate_keys(self):
        self.he.keyGen()
        self.he.rotateKeyGen()

    def generate_relin_keys(self):
        self.he.relinKeyGen()

    def get_public_key(self):
        self.he.save_public_key(tmp_dir.name + "/pub.key")
        with open(tmp_dir.name + "/pub.key", 'rb') as f:
            return f.read()

    def get_relin_key(self):
        self.he.save_relin_key(tmp_dir.name + "/relin.key")
        with open(tmp_dir.name + "/relin.key", 'rb') as f:
            return f.read()

    def load_public_key(self, key):
        with open(tmp_dir.name + "/pub.key", 'wb') as f:
            f.write(key)
        self.he.load_public_key(tmp_dir.name + "/pub.key")

    def load_relin_key(self, key):
        with open(tmp_dir.name + "/relin.key", 'wb') as f:
            f.write(key)
        self.he.load_relin_key(tmp_dir.name + "/relin.key")

    def encode_matrix(self, matrix):
        """Encode a float nD-matrix in a PyPtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded
        Returns
        -------
        matrix
            nD-np.array( dtype=PyPtxt ) with encoded values
        """

        try:
            return np.array(list(map(self.encode, matrix)))
        except TypeError:
            return np.array([self.encode_matrix(m) for m in matrix])

    def decode_matrix(self, matrix):
        """Decode a PyPtxt nD-matrix in a float nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyPtxt )
            matrix to be decoded
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with float values
        """
        try:
            return np.array(list(map(self.decode, matrix)))
        except TypeError:
            return np.array([self.decode_matrix(m) for m in matrix])

    def encrypt_matrix(self, matrix):
        """Encrypt a float nD-matrix in a PyCtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=PyCtxt ) with encrypted values
        """
        try:
            return np.array(list(map(self.encrypt, matrix)))
        except TypeError:
            return np.array([self.encrypt_matrix(m) for m in matrix])



    def decrypt_matrix(self, matrix):
        """Decrypt a PyCtxt nD matrix in a float nD matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyCtxt )
            matrix to be decrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with plain values
        """
        try:
            return np.array(list(map(self.decrypt, matrix)))
        except TypeError:
            return np.array([self.decrypt_matrix(m) for m in matrix])

    def encode_number(self, number):
        return self.encode(number)

    def power(self, number, exp):
        if isinstance(number, np.ndarray):
            raise TypeError
        if exp != 2:
            raise NotImplementedError("Only square")
        s = number * number
        self.he.relinearize(s)
        self.he.rescale_to_next(s)
        return s

    def noise_budget(self, ciphertext):
        return None

class LinearLayer:
    """
    A class used to represent a linear (fully connected) layer
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to encode weights and bias
    weights : np.array( dtype=PyPtxt )
        Weights of the layer, in form
        [out_features, in_features]
    bias : np.array( dtype=PyPtxt ), default=None
        Biases of the layer, 1-D array


    Methods
    -------
    __init__(self, HE, weights, bias=None)
        Constructor of the layer, bias is set to None if not provided.
    __call__(self, t)
        Execute che linear operation on a flattened input, t, in the form
            [n_images, in_features], 2D-np.array( dtype=PyCtxt )
        using weights and biases of the layer.
        returns the result in the form
            [n_images, out_features], 2D-np.array( dtype=PtCtxt )
    """

    def __init__(self, HE, weights, bias=None):
        self.HE = HE
        self.weights = HE.encode_matrix(weights)
        self.bias = bias

        if bias is not None:
            self.bias = HE.encode_matrix(bias)

    def __call__(self, t):
        start = time.time()
        result = np.array([[np.sum(image * row) for row in self.weights] for image in t])

        if self.bias is not None:
            result = np.array([row + self.bias for row in result])
        print("linear time: " + str(time.time()-start))
        return result

class Identity:
    def __call__(self, image):
        return image
    

# Square acivation layer
class SquareLayer:
    """
    A class used to represent a layer which performs the square
    of the values in a nD-matrix.

    ...

    Notes
    -----
    The values inside the matrix are squared, not the matrix themself

    Attributes
    ----------

    HE: Pyfhel
        Pyfhel object

    Methods
    -------
    __init__(self, HE)
        Constructor of the layer.
    __call__(self, image)
        Executes the square of the input matrix.
    """
    def __init__(self, HE):
        self.HE = HE

    def __call__(self, image):
        start = time.time()
        val = square(self.HE, image)
        print("square time: " + str(time.time()-start))
        return val


def square(HE, image):
    """Execute the square operation, given a batch of images,

        Parameters
        ----------
        HE : Pyfhel object
        image : np.array( dtype=PyCtxt )
            Encrypted nD matrix to square

        Returns
        -------
        result : np.array( dtype=PtCtxt )
            Encrypted result of the square operation
        """

    try:
        return np.array(list(map(lambda x: HE.power(x, 2), image)))
    except TypeError:
        return np.array([square(HE, m) for m in image])

# Sigmoid acivation layer
class SigmoidLayer:
    def __init__(self, HE):
        self.HE = HE

    def __call__(self, image):
        return sigmoid(self.HE, image)

def sigmoid(HE, image):
    try:
        return np.array(list(map(lambda x: compute_sigmoid(HE, x), image)))
    except TypeError:
        return np.array([sigmoid(HE, m) for m in image])

def compute_sigmoid(HE, x):
    # 0.5+0.197x-0.004x^3
    constant = HE.encode(0.5)
    coeff1 = HE.encode(0.197)
    coeff2 = HE.encode(0.004)
    
    t2 = HE.power(x, 2)
    t3 = t2 * x
    HE.he.relinearize(t3)
    HE.he.rescale_to_next(t3)

    term2 = coeff1 * x
    HE.he.rescale_to_next(term2)
    term3 = coeff2 * t3
    HE.he.rescale_to_next(term3)

    x = constant + term2 - term3

    return x

# Tanh acivation layer
class TanhLayer:
    def __init__(self, HE):
        self.HE = HE

    def __call__(self, image):
        if image.shape[1] != 5:
            start = time.time()
            val = tanh(self.HE, image)
            print("tanh time: " + str(time.time()-start))
            return val
        else:
            return image

def tanh(HE, image):
    try:
        return np.array(list(map(lambda x: compute_tanh(HE, x), image)))
    except TypeError:
        return np.array([tanh(HE, m) for m in image])

def compute_tanh(HE, x):
    coeff1 = HE.encode(0.333)
    coeff2 = HE.encode(0.133)
    
    t2 = HE.power(x, 2)
    t3 = t2 * x
    HE.he.relinearize(t3)
    HE.he.rescale_to_next(t3)

    t5 = t3*t2
    HE.he.relinearize(t5)
    HE.he.rescale_to_next(t5)

    term2 = coeff1 * t3
    HE.he.rescale_to_next(term2)
    term3 = coeff2 * t5
    HE.he.rescale_to_next(term3)

    x = x - term2 + term3
    return x

# flatten
class FlattenLayer:
    """
    A class used to represent a layer which performs a
    flattening operation.
    ...

    Attributes
    ----------

    length: int
        second dimension of the output matrix

    Methods
    -------
    __init__(self, length)
        Constructor of the layer.
    __call__(self, t)
        Taken an input tensor in the form
            [n_images, n_layers, y, x]
        Reshapes it in the form
            [n_images, length]
    """

    def __call__(self, image):
        start = time.time()
        dimension = image.shape
        try:
            val = image.reshape(dimension[0], dimension[1]*dimension[2]*dimension[3])
            print("flatten time: " + str(time.time()-start))
            return val
        except:
            return image.reshape(dimension[0], dimension[1]*dimension[2])

class AveragePoolLayerTest:
    """
    A class used to represent a layer which performs an average pool operation
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to perform the pool operation
    kernel_size: int
        Size of the square kernel
    stride : int
        Stride of the pool operaiton

    Methods
    -------
    __init__(self, HE, kernel_size, stride)
        Constructor of the layer.
    __call__(self, t)
        Execute che average pooling operation on a batch of images, t, in the form
            [n_images, n_layers, y, x]
        using kernel size and strides of the layer.
    """
    def __init__(self, HE, kernel_size, stride, padding=(0, 0)):
        self.HE = HE
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def __call__(self, t):
        start = time.time()
        if self.padding != (0,0):
            t = apply_padding(t, self.padding)
        result = np.array([[_avgtest(self.HE, layer, self.kernel_size, self.stride) for layer in image] for image in t])
        print("avgpool time: " + str(time.time()-start))
        return result


def _avgtest(HE, image, kernel_size, stride):
    """Execute an average pooling operation given an 2D-image,
        a kernel-size and a stride.


    Parameters
    ----------
    HE: PYfhel object
    image : np.array( dtype=PyCtxt )
        Encrypted image to execute the pooling, in the form
        [y, x]
    kernel_size : (int, int)
        size of the kernel (y, x)
    stride : (int, int)
        stride (y, x)
    Returns
    -------
    result : np.array( dtype=PtCtxt )
        Encrypted result of the pooling, in the form
        [y, x]
    """
    # print("inside")
    x_s = stride[1]
    y_s = stride[0]

    x_k = kernel_size[1]
    y_k = kernel_size[0]

    x_d = len(image[0])
    y_d = len(image)

    x_o = ((x_d - x_k) // x_s) + 1
    y_o = ((y_d - y_k) // y_s) + 1

    denom = x_k * y_k
    denom = 0.5 + 0.25*denom - 0.125*denom**3
    denominator = HE.encode(1 / denom)

    def get_submatrix(matrix, x, y):
        index_row = y * y_s
        index_column = x * x_s
        return matrix[index_row: index_row + y_k, index_column: index_column + x_k]
    
    term1 = HE.encode(0.5)
    coeff1 = HE.encode(0.25)
    coeff2 = HE.encode(0.125)

    bigarray = []
    for y in range(0, y_o):
        array = []
        for x in range(0, x_o):
            result = np.sum(get_submatrix(image, x, y))
            t3 = HE.power(result, 2) * result
            HE.he.relinearize(t3)
            HE.he.rescale_to_next(t3)

            term2 = coeff1 * result
            HE.he.rescale_to_next(term2)
            term3 = coeff2 * t3
            HE.he.rescale_to_next(term3)

            result = term1 + term2 - term3

            result = result * denominator
            HE.he.rescale_to_next(result)

            if (result.size()>2):
                HE.he.relinearize(result)

            array.append(result)
        bigarray.append(array)
    return bigarray

########################## Actual AvgPool2d ##########################

def apply_padding(t, padding):
    """Execute a padding operation given a batch of images in the form
        [n_image, n_layer, y, x]
       After the execution, the result will be in the form
        [n_image, n_layer, y+padding, x+padding]
       The element in the new rows/column will be zero.
       Due to Pyfhel limits, a sum/product between two PyPtxt can't be done.
       This leads to the need of having a PyCtxt which has to be zero if decrypted: this is done by
       subtracting an arbitrary value to itself.

    Parameters
    ----------
    t: np.array( dtype=PyCtxt )
        Encrypted image to execute the padding on, in the form
        [n_images, n_layer, y, x]
    padding: (int, int)

    Returns
    -------
    result : np.array( dtype=PyCtxt )
        Encrypted result of the padding, in the form
        [n_images, n_layer, y+padding, x+padding]
    """

    y_p = padding[0]
    x_p = padding[1]
    zero = t[0][0][y_p+1][x_p+1] - t[0][0][y_p+1][x_p+1]
    return [[np.pad(mat, ((y_p, y_p), (x_p, x_p)), 'constant', constant_values=zero) for mat in layer] for layer in t]

# convolutional
class Conv2d:
    """
    A class used to represent a convolutional layer
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to encode weights and bias
    weights : np.array( dtype=PyPtxt )
        Weights of the layer, aka filters in form
        [n_filters, n_layers, y, x]
    stride : (int, int)
        Stride (y, x)
    padding : (int, int)
        Padding (y, x)
    bias : np.array( dtype=PyPtxt ), default=None
        Biases of the layer, 1-D array


    Methods
    -------
    __init__(self, HE, weights, x_stride, y_stride, bias=None)
        Constructor of the layer, bias is set to None if not provided.
    __call__(self, t)
        Execute che convolution operation on a batch of images, t, in the form
            [n_images, n_layers, y, x]
        using weights, biases and strides of the layer.
    """

    def __init__(self, HE, HE_Client, weights, stride, padding=(0, 0), bias=None):
        self.HE = HE
        self.HE_Client = HE_Client
        self.weights = weights
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def __call__(self, t):
        start = time.time()
        if self.padding != (0,0):
            t = apply_padding(t, self.padding)

        result = np.array([[np.sum([convolute2d(self.HE, image_layer, filter_layer, self.stride)
                                    for image_layer, filter_layer in zip(image, _filter)], axis=0)
                            for _filter in self.weights]
                           for image in t])

        if self.bias is not None:
            result = np.array([[layer + bias for layer, bias in zip(image, self.bias)] for image in result])
        print("conv2d time: " + str(time.time()-start))
        return result

def convolute2d(HE, image, filter_matrix, stride):
    """Execute a convolution operation given an 2D-image, a 2D-filter
    and related strides.


    Parameters
    ----------
    image : np.array( dtype=PyCtxt )
        Encrypted image to execute the convolution on, in the form
        [y, x]
    filter_matrix : np.array( dtype=PyPtxt )
        Encoded weights to use in the convolution, in the form
        [y, x]
    stride : (int, int)
        Stride

    Returns
    -------
    result : np.array( dtype=PtCtxt )
        Encrypted result of the convolution, in the form
        [y, x]
    """
    x_d = len(image[0])
    y_d = len(image)
    x_f = len(filter_matrix[0])
    y_f = len(filter_matrix)

    y_stride = stride[0]
    x_stride = stride[1]

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    def get_submatrix(matrix, x, y):
        index_row = y * y_stride
        index_column = x * x_stride
        return matrix[index_row: index_row + y_f, index_column: index_column + x_f]
    
    return np.array([[np.sum(get_submatrix(image, x, y) * filter_matrix) for x in range(0, x_o)] for y in range(0, y_o)])

tmp_dir = tempfile.TemporaryDirectory()

class Sequential:
    """
    Class which mimics PyTorch Sequential models.
    This class will be used as a container for the
    PyCrCNN layers.
    """
    layers = []

    """
    Given a PyTorch sequential model, create the corresponding PyCrCNN model.
    """
    def __init__(self, HE_Server, HE_Client, model):
        self.HE = HE_Server
        self.HE_Client = HE_Client
        
        def conv2d(layer):
            if layer.bias is None:
                bias = None
            else:
                bias = layer.bias.detach().numpy()
            
            return Conv2d(self.HE, self.HE_Client, weights=layer.weight.detach().numpy(),
                          stride=layer.stride,
                          padding=layer.padding,
                          bias=bias,)

        def lin_layer(layer):
            if layer.bias is None:
                bias = None
            else:
                bias = layer.bias.detach().numpy()
            return LinearLayer(self.HE, layer.weight.detach().numpy(),
                               bias,)

        def avg_pool_layer_test(layer):
            kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size,
                                                                               int) else layer.kernel_size
            stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
            padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding
            return AveragePoolLayerTest(self.HE, kernel_size, stride, padding)

        def flatten_layer(layer):
            return FlattenLayer()
                
        def none(layer):
            return Identity()

        
        # Maps every PyTorch layer type to the correct builder
        options = {"Conv": conv2d,
                   "Line": lin_layer,
                   "Flat": flatten_layer,
                   "AvgP": avg_pool_layer_test,
                   "Sigm": none,
                   }

        # because model is pretrained and saved in PyTorch, model[i] = ith layer of model, and each layer has layer.weights, layer.bias attributes inherently
        self.layers = [options[str(layer)[0:4]](layer) for layer in model]

    def __call__(self, x, debug=False):
        for layer in self.layers:
            x = layer(x)

        return x

class HE:
    def generate_keys(self):
        pass

    def generate_relin_keys(self):
        pass

    def get_public_key(self):
        pass
    
    def generate_switching_key(self):
        pass

    def get_relin_key(self):
        pass

    def load_public_key(self, key):
        pass

    def load_relin_key(self, key):
        pass

    def encode_matrix(self, matrix):
        """Encode a matrix in a plaintext HE nD-matrix.

        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded

        Returns
        -------
        matrix
            nD-np.array with encoded values
        """
        pass

    def decode_matrix(self, matrix):
        pass

    def encrypt_matrix(self, matrix):
        pass

    def decrypt_matrix(self, matrix):
        pass

    def encode_number(self, number):
        pass

    def power(self, number, exp):
        pass

    def noise_budget(self, ciphertext):
        pass

############################## Define LeNet ##############################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        # 0.5+0.25x-0.125x^{3}
        return 0.5 + 0.25*t - 0.125*t**3

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.1307], std=[0.3081])])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train_net(network, epochs, device):
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    for _ in range(epochs):

        total_loss = 0
        total_correct = 0

        for batch in train_loader: # Get Batch
            images, labels = batch 
            images, labels = images.to(device), labels.to(device)

            preds = network(images) # Pass Batch
            loss = F.cross_entropy(preds, labels) # Calculate Loss

            optimizer.zero_grad()
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
   
def test_net(network, device):
    network.eval()
    total_loss = 0
    total_correct = 0
    
    with torch.no_grad():
        for batch in test_loader: # Get Batch
            images, labels = batch 
            images, labels = images.to(device), labels.to(device)

            preds = network(images) # Pass Batch
            loss = F.cross_entropy(preds, labels) # Calculate Loss

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        accuracy = round(100. * (total_correct / len(test_loader.dataset)), 4)

    print(accuracy)
    return total_correct / len(test_loader.dataset)

############################## Train and test LeNet ##############################

train = True # If set to false, it will load models previously trained and saved.

experiments = 1

if train:
    approx_accuracies = []
    for i in range(0, experiments):
        LeNet1_Approx = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5),
            nn.AvgPool2d(kernel_size=3),
            Sigmoid(),

            nn.Flatten(),

            nn.Linear(256, 10),
        )
        
        LeNet1_Approx.to(device)
        train_net(LeNet1_Approx, 10, device)
        acc = test_net(LeNet1_Approx, device)
        approx_accuracies.append(acc)
        
    torch.save(LeNet1_Approx, "avg-5-sigmoid-swap-k3.pt")

n_mults = 8
m = 16384
scale_power = 30

encryption_parameters = {
    'm': m,                      # For CKKS, n/2 values can be encoded in a single ciphertext
    'scale': 2**scale_power,                 # Each multiplication grows the final scale
    'qi': [31]+ [scale_power]*n_mults +[31]  # One intermdiate for each multiplication
}

start_time = time.time()

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
images, labels = next(iter(test_loader))

HE_Client = CKKSPyfhel(**encryption_parameters)
HE_Client.generate_keys()
HE_Client.generate_relin_keys()

public_key = HE_Client.get_public_key()
relin_key  = HE_Client.get_relin_key()

lenet_1_approx = torch.load("avg-5-sigmoid-swap-k3.pt", map_location="cpu")
HE_Server = CKKSPyfhel(**encryption_parameters)
HE_Server.load_public_key(public_key)
HE_Server.load_relin_key(relin_key)

requested_time = round(time.time() - start_time, 2)
print(f"\nThe context generation requested {requested_time} seconds.")

start_time = time.time()
lenet1_approx_encoded = Sequential(HE_Server, HE_Client, lenet_1_approx)
requested_time = round(time.time() - start_time, 2)
print(f"\nThe model encoding requested {requested_time} seconds.")

difference = 0
num_correct = 0
total_time = 0
for i in range(100):
    
    sample_image = images[i]   
    sample_label = labels[i]

    encrypted_image = HE_Client.encrypt_matrix(sample_image.unsqueeze(0).numpy())

    requested_time = round(time.time() - start_time, 2)
    print(f"\nThe image encryption requested {requested_time} seconds.")

    with torch.no_grad():
        expected_output = lenet_1_approx(sample_image.unsqueeze(0))

    start_time = time.time()
    encrypted_output = lenet1_approx_encoded(encrypted_image, debug=False)
    requested_time = round(time.time() - start_time, 2)
    total_time += requested_time
    print(f"\nThe encrypted processing of one image requested {requested_time} seconds.")

    start_time = time.time()
    result = HE_Client.decrypt_matrix(encrypted_output)
    requested_time = round(time.time() - start_time, 2)
    
    print(f"\nThe decryption of one result requested {requested_time} seconds.")

    result_tensor = torch.tensor(result)
    if torch.argmax(result_tensor, dim=1)[0] == sample_label:
        num_correct += 1
    
    print("num correct so far:" +str(num_correct))
    print("out of: " + str(i+1))
    difference += expected_output.numpy() - result

print(f"\nTotal num correct is: ")
print(num_correct)

print("total time is " + str(total_time))

print("avg time is " + str(total_time/100))

print("avg difference is: " + str(difference/100))
