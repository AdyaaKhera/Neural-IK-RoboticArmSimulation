from forward_kinematics import fdotk, link1, link2
import random, csv, math
with open("data/dataset.csv", "w", newline = "") as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y", "theta1", "theta2"]) #writing the header
    
    count = 0
    attempts = 0
    maxattempts = 50000
    while count < 10000 and attempts < maxattempts:
        #generating dataset with predefined restraints
        theta1 = random.uniform(0, 90) 
        theta2 = random.uniform(-90, 0)

        if theta1 < theta2 or theta1 + theta2 > 150: #ensuring natural elbow position dataset discarding the unnatural ones
            continue

        else:
            #computing elbow position
            theta1_rad = math.radians(theta1)
            x1 = link1 * math.cos(theta1_rad)
            y1 = link1 * math.sin(theta1_rad)

            if y1 < 0:
                continue

            x, y = fdotk(theta1, theta2)

            if y>= 0:
                writer.writerow([x, y, theta1, theta2])
                count +=1
        attempts += 1
    
    if count < 10000:
        print(f"only {count} samples were generated after {attempts} attempts.")