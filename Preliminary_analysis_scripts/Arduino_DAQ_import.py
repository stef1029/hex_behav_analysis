import json


class Arduino_DAQ_Import:

    def __init__(self, arduino_daq_data_path, pulses):
        self.arduino_daq_data_path = arduino_daq_data_path
        self.pulses = pulses

        self.ingest_arduino_daq_data()
    
    def ingest_arduino_daq_data(self):
        with open(self.arduino_daq_data_path, 'r') as f:
            self.data = json.load(f)

        self.messages = self.data["messages"]
        self.message_ids = [message[0] for message in self.messages]
        self.message_data = [message[1] for message in self.messages]

        # 32 bit message, and read from last digit to first digit, so needs these blank1 and blank2 channels to be ignored but needed to avoid index errors when reading.
        self.channel_indices = ("SPOT2", "SPOT3", "SPOT4", "SPOT5", "SPOT6", "SPOT1", "SENSOR6", "SENSOR1", "SENSOR5", "SENSOR2", "SENSOR4", "SENSOR3", "BUZZER4", "LED_3", "LED_4", "BUZZER3", "BUZZER5", "LED_2", "LED_5", "BUZZER2", "BUZZER6", "LED_1", "LED_6", "BUZZER1", "VALVE4", "VALVE3", "VALVE5", "VALVE2", "VALVE6", "VALVE1", "GO_CUE", "NOGO_CUE")

        self.channel_data = {channel: [] for channel in self.channel_indices}
        self.channel_data["message_ids"] = self.message_ids
        
        for i, message in enumerate(self.message_data):
            message_id = self.message_ids[i]
            if message_id > len(self.pulses):
                continue
            else:
                for j, digit in enumerate(message):
                    channel_title = self.channel_indices[-(j+1)]
                    self.channel_data[channel_title].append(digit)
        
        
if __name__ == "__main__":
    import_test = Arduino_DAQ_Import(r"Y:\Private_Lab\Stefan\231205_213113_test\231205_213113_test-ArduinoDAQ.json").channel_data

    print(import_test["SENSOR1"][:100])