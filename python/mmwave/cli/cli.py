import serial
from .cmd import encode_cmd

PROMPT = 'mmwDemo:/>'
DONE_FLAG = 'Done'
ERROR_FLAG = 'Error'


class MMWaveCLI:
    def __init__(self, user_port: str, baud_rate: int = 115200, flush: bool = True):
        self.stream = serial.Serial(user_port, baud_rate)
        assert self.stream is not None, "Unable to connect to control port"

        if flush and self.stream.isOpen():
            self.stream.close()
            self.stream.open()

    def close(self):
        self.stream.close()

    def send_cmd(self, cmd, *args):
        cmd = encode_cmd(cmd, *args)
        raw_cmd = bytes(cmd, 'utf-8')

        self.stream.write(raw_cmd + b'\n')
        lines = self.stream.read_until(PROMPT.encode())
        lines = lines.split(b'\n')

        prompt_line = lines[-1].decode().strip()
        assert prompt_line == PROMPT

        if prompt_line != PROMPT:
            raise BaseException(f"Timed out, got '{prompt_line}' instead of prompt")

        result_line = lines[-2].decode().strip()
        details_lines = lines[1:-2]
        if result_line.startswith(ERROR_FLAG):
            error_code = int(result_line.split(" ")[-1])
            details = "\n".join(line.decode().strip() for line in details_lines)
            raise BaseException(error_code, details)
        elif result_line != DONE_FLAG:
            raise BaseException(f"Got unexpected result '{result_line}'")

        return details_lines
