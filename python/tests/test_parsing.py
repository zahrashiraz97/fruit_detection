from mmwave.cli.cmds import ChannelCfg
from mmwave.cli import encode_cmd, parse_cmd

import unittest


class TestParsing(unittest.TestCase):

    def test_registered(self):
        expected = ChannelCfg(15, 1, 0)

        # Unparsed str
        cmd = parse_cmd("channelCfg 15 1 0")
        self.assertEqual(cmd, expected)

        # Unparsed typed sequence
        cmd = parse_cmd("channelCfg", 15, 1, 0)
        self.assertEqual(cmd, expected)

        # Unparsed untyped sequence
        cmd = parse_cmd("channelCfg", "15", "1", "0")
        self.assertEqual(cmd, expected)

        # Parsed unregistered format
        cmd = ("channelCfg", (15, 1, 0))
        cmd = parse_cmd(cmd)
        self.assertEqual(cmd, expected)

        # Parsed registered format
        cmd = ChannelCfg(15, 1, 0)
        cmd = parse_cmd(cmd)
        self.assertEqual(cmd, expected)

    def test_unregistered(self):
        expected = ("someRandomCmd", ("0", "1", "2"))

        # Unparsed str
        cmd = parse_cmd("someRandomCmd 0 1 2")
        self.assertEqual(cmd, expected)

        # Unparsed typed sequence
        cmd = parse_cmd("someRandomCmd", 0, 1, 2)
        self.assertNotEqual(cmd, expected)

        # Unparsed untyped sequence
        cmd = parse_cmd("someRandomCmd", "0", "1", "2")
        self.assertEqual(cmd, expected)


class TestEncoding(unittest.TestCase):
    def test_registered(self):
        expected = "channelCfg 15 1 0"

        # Unparsed str
        encoded = encode_cmd(expected)
        self.assertEqual(encoded, expected)

        # Unparsed sequence
        encoded = encode_cmd("channelCfg", 15, 1, 0)
        self.assertEqual(encoded, expected)

        # Unregistered
        cmd = ("channelCfg", (15, 1, 0))
        encoded = encode_cmd(cmd)
        self.assertEqual(encoded, expected)

        # Registered
        cmd = ChannelCfg(15, 1, 0)
        encoded = encode_cmd(cmd)
        self.assertEqual(encoded, expected)


if __name__ == '__main__':
    unittest.main()
