import struct


class FaultInjectorEngine:

    @staticmethod
    def float32_stuck_at(float_number, position, stuck_at_value):
        """
        Performs a stuck-at fault on the bit in a given position of the input float32
        :param float_number: the float32 to which apply the stuck-at fault
        :param position: the position (0-31) of the bit that will be affected by the fault
        :param stuck_at_value: the value to set the bit to
        :return: A float32 containing the number affected by the fault
        """
        float_list = []
        a = struct.pack('!f', float_number)
        b = struct.pack('!I', (2 ** position))
        for ba, bb in zip(a, b):
            if stuck_at_value == 1:
                float_list.append(ba | bb)
            else:
                float_list.append(ba & (255 - bb))

        return struct.unpack('!f', bytes(float_list))[0]

    @staticmethod
    def float32_bit_flip(float_number, position):
        """
        Performs a bit-flip on the bit in a given position of the input float32
        :param float_number: the float32 to which apply the bit-flip
        :param position: the position (0-31) of the bit that will be flipped
        :return: A float32 containing the number with the bit flipped
        """
        float_list = []
        a = struct.pack('!f', float_number)
        b = struct.pack('!I', (2 ** position))
        for ba, bb in zip(a, b):
            float_list.append(ba ^ bb)

        return struct.unpack('!f', bytes(float_list))[0]
