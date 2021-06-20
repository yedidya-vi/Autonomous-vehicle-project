import cv2
import numpy as np
import math


class detect:
  def __init__(self, img):
    _ , self.img = img
    self.height = self.img.shape[0]
    self.width = self.img.shape[1]

  def process(self):
      region_of_interest_vertices = [
          (0, self.height),
          (200, self.height - 400),
          (self.width / 2, self.height / 4),
          (self.width - 200, self.height - 400),
          (self.width, self.height)
      ]
      hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
      yellow_lower = np.array([20, 40, 40], dtype=np.uint8)
      yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
      mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
      edges = cv2.Canny(mask, 100, 100)
      mask = np.zeros_like(edges)
      cv2.fillPoly(mask,  np.array([region_of_interest_vertices], np.int32), 255)
      cropped_image = cv2.bitwise_and(edges, mask)
      cv2.imshow('mask', cropped_image)
      lines = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=10)
      return lines

  def make_points(self, line):
      height, width, _ = frame.shape
      slope, intercept = line
      y1 = height  # bottom of the frame
      y2 = int(y1 * 1 / 3)  # make points from middle of the frame down

        # bound the coordinates within the frame
      x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
      x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
      return [[x1, y1, x2, y2]]

  def AverageLines (self, lines):
      #lane_lines = []
      if lines is None:
          return self.img

      left_fit = []
      right_fit = []

      for line_segment in lines:
          for x1, y1, x2, y2 in line_segment:
              if x1 == x2:
                  continue
              if abs((y2 - y1) / (x2 - x1)) < 0.09:
                  continue
              if abs((y2 - y1) / (x2 - x1)) < 0.6:
                  if np.sqrt(np.square(abs(y2 - y1)) + np.square(abs(x2 - x1))) < 200:
                      continue
              if np.sqrt(np.square(abs(y2 - y1)) + np.square(abs(x2 - x1))) < 70:
                  continue
              fit = np.polyfit((x1, x2), (y1, y2), 1)
              slope = fit[0]
              intercept = fit[1]
              if slope < 0:
                  left_fit.append((slope, intercept))
                  continue
              else:
                  right_fit.append((slope, intercept))

      right_fit_average = np.average(right_fit, axis=0)
      if right_fit_average is None:
          return None

      if len(right_fit) > 0:
          x1, y1, x2, y2 = (detect.make_points(self, right_fit_average))
      cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)

      return [x1, y1, x2, y2]

  def make_points(self,line):
      slope, intercept = line
      y1 = self.height  # bottom of the frame
      y2 = int(y1 * 1 / 3)  # make points from middle of the frame down

      # bound the coordinates within the frame
      x1 = max(-self.width, min(2 * self.width, int((y1 - intercept) / slope)))
      x2 = max(-self.width, min(2 * self.width, int((y2 - intercept) / slope)))
      return [x1, y1, x2, y2]

  def middleLine(self, avarage_line):

      if avarage_line == []:
          return

      x1, _, x2, _ = avarage_line
      x_offset = x2 - x1
      y_offset = int(self.height / 3)
      angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
      angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
      steering_angle = angle_to_mid_deg + 158  # this is the steering angle needed by picar front wheel
      steering_angle_radian = steering_angle / 180.0 * math.pi
      x1 = int(self.width / 2)
      y1 = self.height
      x2 = int(x1 - self.height / 2 / math.tan(steering_angle_radian))
      y2 = int(self.height / 2)
      cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 5)
      xa, ya ,xb ,yb = x1, y1, x1, y1 - 300
      cv2.line(self.img, (xa, ya), (xb, yb), (0, 255, 0), 5)

      return steering_angle - 90

  def DrawAngle(self, steering_angle):
      angle = steering_angle
      txt = 'The deviation angle is: ' + str(angle)
      angleText = cv2.putText(self.img, txt, (int((self.width / 2) - 300), 100),
                              cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)


cap = cv2.VideoCapture('video_name')
while (cap.isOpened()):
    try:
        frame = detect(cap.read())
        cv2.imshow('frame1q', frame.img)
        lines = frame.process()
        avarage_line = frame.AverageLines(lines)
        steering_angle = frame.middleLine(avarage_line)
        if abs(steering_angle) > 30:
            continue
        frame.DrawAngle(steering_angle)
        cv2.imshow('frame', frame.img)
        if cv2.waitKey(5) == ord('q'):
            break
    except:
        continue

cap.release()
cv2.destroyAllWindows()