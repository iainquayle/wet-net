#9690016f-ca9f-11ed-818a-a0369f818cc4

import requests
import os

noaa_url_fragment_list = [("https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/AirMass/2023", "_GOES16-ABI-FD-AirMass-1808x1808.jpg"), #10 airmass min interval, XXX0
	("https://cdn.star.nesdis.noaa.gov/GOES16/GLM/FD/EXTENT3/2023", "_GOES16-GLM-FD-EXTENT3-1808x1808.jpg"), #5 surface colour and lightening min interval, XXX1-6 
	("https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/DayNightCloudMicroCombo/2023", "_GOES16-ABI-FD-DayNightCloudMicroCombo-1808x1808.jpg"), #10 microclouds min interval, XXX0 
	("https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/Sandwich/2023", "_GOES16-ABI-FD-Sandwich-1808x1808.jpg"),] #10 sandwich, shows convection storms, XXX0

def str_day(day):
	return str(day).zfill(3)
def str_hour(hour):
	return str(hour).zfill(2)
def str_min(min):
	return str(min).zfill(2)

class URLFormatter():
	def __init__(self, name, interval, url_function, extension = ".jpg"):
		self.name = name
		self.extension = extension
		self.url_function = url_function
		self.interval = interval
	def format(self, day, hour, minute):
		day = str_day(day)
		hour = str_hour(hour) 
		minute = str_min(minute)
		return self.url_function(day, hour, minute)

noaa_urls = [URLFormatter("airmass", 10, lambda d, h, m: "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/AirMass/2023{}{}{}_GOES16-ABI-FD-AirMass-1808x1808.jpg".format(d, h, m)),
	URLFormatter("surfacelightening", 5, lambda d, h, m: "https://cdn.star.nesdis.noaa.gov/GOES16/GLM/FD/EXTENT3/2023{}{}{}_GOES16-GLM-FD-EXTENT3-1808x1808.jpg".format(d, h, m)), 
	URLFormatter("microcloud", 10, lambda d, h, m: "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/DayNightCloudMicroCombo/2023{}{}{}_GOES16-ABI-FD-DayNightCloudMicroCombo-1808x1808.jpg".format(d, h, m)),
	URLFormatter("sandwich", 10, lambda d, h, m: "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/Sandwich/2023{}{}{}_GOES16-ABI-FD-Sandwich-1808x1808.jpg".format(d, h, m))]

def download_batch(url_formatter, save_path, start_day, end_day, start_hour, end_hour, start_minute, end_minute):
	day = start_day
	hour = start_hour
	minute = start_minute
	while day < end_day or hour < end_hour or minute < end_minute:
		minute %= 60 
		hour %= 24
		result = requests.get(url_formatter.format(day, hour, minute))
		if  result.ok:
			new_path = "{}/{}/{}{}{}{}".format(save_path, url_formatter.name, str_day(day), str_hour(hour), str_min(minute), url_formatter.extension)
			os.makedirs(os.path.dirname(new_path), exist_ok=True)
			with open(new_path, 'wb') as f:
				f.write(result.content)
			print(url_formatter.name + " : " + str_day(day) + str_hour(hour) + str_min(minute))
		else: 
			print("url failed")
		minute += url_formatter.interval
		if minute % 60 < minute:
			hour += 1
		if hour % 24 < hour:
			day += 1

#print(noaa_urls[0].format(82, 1750))
download_batch(noaa_urls[3], './data', 82, 92, 19, 20, 30, 00)
