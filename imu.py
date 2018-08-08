class IMU:
    iteration = 1
    t_prev = 0

    jerk = [0,0,0]
    accel = [0,0,0]

    acceleration = [0,0,0]
    velocity = [0,0,0]
    location = [0,0,0]

    quaternion = [1,0,0,0]

    sliding_window = 50
    magnitudes = [0]
    velocities_drifted = [[0,0,0]]

    def step(self):
        # Main method:
        # uses inertial sensors to estimate the xyz positions and xyz rotations 
        # calculates quaternion to remove effect of gravity and converts to a global reference frame
        #returns 6element vector [forward_back,right_left,up_down,roll,pitch,yaw]
        ss = self.sensors()
        #ss['accelerometer'] = self.removeGravity(ss['accelerometer'])
        self.quaternion_update(ss['accelerometer'],ss['gyroscope'],ss['compass'])
        self.location_update(self.global_acceleration(ss['accelerometer']))        
        return self.location + self._angles(self.quaternion)

    def sensors(self):
        # reads in a single timestep of sensor data stored on files (simulating live sensor data)
        # calculates time delay between sensor readings (self.dt)
        # returns: dictionary containing:
        #   - acceleration: xyz (accelerations along axes relative to sensor)
        #   - gyroscope: xyz (angular veclocities along axes relative to sensor)
        #   - compass: xyz (magnetic strength along axes relative to sensor)
        sensor_data = {}
        dt = 0
        while dt == 0:
            self.iteration += 1
            for sensor in ('accelerometer','gyroscope','compass'):
                with open('data/{}.csv'.format(sensor)) as s:
                    for _ in range(self.iteration):
                        data = s.readline()
                    time,x,y,z = data.strip().split(',')
                sensor_data[sensor] = [float(x),float(y),float(z)]
            dt = (int(time) - self.t_prev) / 1000 #ms
            self.t_prev = int(time) 
        self.dt = dt
        return sensor_data

    '''
    def removeGravity(self,acceleration):
        # differentiate acceleration to find jerk (rate of accel)
        # filter gradual changes in acceleration to remove gravity
        # then re-integrate to get acceleration without gravity
        jerk = [self._differentiate(a,a_prev) for a,a_prev in zip(acceleration,self.acceleration)]
        
        dAx = min(10,max(-10,self._integrate(jerk[0],self.jerk[0])))
        dAy = min(10,max(-10,self._integrate(jerk[1],self.jerk[1])))
        dAz = min(10,max(-10,self._integrate(jerk[2],self.jerk[2])))

        accel = [self.accel[0] + dAx, self.accel[1] + dAy, self.accel[2] + dAz]

        self.jerk = jerk
        self.accel = accel
        return accel
    '''

    def location_update(self,acceleration):
        # updates the current location of device in 3D space 
        # by integrating acceleration values twice
        # and compensating for drift using a light-weight algorithm                 
        dVx = self._integrate(acceleration[0],self.acceleration[0])
        dVy = self._integrate(acceleration[1],self.acceleration[1])
        dVz = self._integrate(acceleration[2],self.acceleration[2])

        vel_prev = self.velocities_drifted[-1]
        velocity_drifted = [vel_prev[0] + dVx, vel_prev[1] + dVy, vel_prev[2] + dVz]        
        velocity = self._drift(velocity_drifted,self._minima())
        
        dLx = self._integrate(velocity[0],self.velocity[0])
        dLy = self._integrate(velocity[1],self.velocity[1])
        dLz = self._integrate(velocity[2],self.velocity[2])

        location = [self.location[0] + dLx, self.location[1] + dLy, self.location[2] + dLz]

        magnitude = self._norm(acceleration)# sum([abs(a) for a in acceleration])
        self.magnitudes.append(magnitude) #add magnitude of acceleration to cache 
        self.magnitudes = self.magnitudes[-self.sliding_window:]
        self.velocities_drifted.append(velocity_drifted) 
        self.velocities_drifted = self.velocities_drifted[-self.sliding_window:]

        self.acceleration = acceleration #update prev acceleration
        self.velocity = velocity #update accumulated velocity (corrected)
        self.location = location #update accumulated position

    def global_acceleration(self,accelerometer):
        #will convert raw accelerations into a global reference frame (North-South,East-West,Up-Down) using a rotation matrix
        # [acc_north, acc_east, acc_up] = [rotation_matrix ^-1 ] @ [acc_x,acc_y,acc_z]
        rotation_matrix = self._matrix(self.quaternion)
        acceleration = self._dot(self._inverse(rotation_matrix), [accelerometer])
        return self._q_minus_q(acceleration[0],[0,0,9.8]) #remove gravity from downward component
        #gravity = self._gravity(acceleration[0],self.quaternion)
        #return [a-g for a,g in zip(acceleration[0],gravity)] #remove component of gravity

    def quaternion_update(self,accelerometer,gyroscope,magnetometer):
        #ahrs = attitude heading reference system
        # sensor fusion to calculate a rotation vector 
        # (algorithm: madwicks quaternion update)
        
        #normalise sensor readings
        accelerometer = self._div(accelerometer,self._norm(accelerometer))
        magnetometer = self._div(magnetometer,self._norm(magnetometer)) 

        #calculations according to madgwicks quaternion update
        q = self.quaternion
        h = self._q_mul_q([0.] + magnetometer , self._conj(q))
        h = self._q_mul_q(q , h)        
        b = [0., self._norm(h[1:3]), 0., h[3]]
        sensor_fusion = [
            2* (q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2* (q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2* (.5 - q[1] **2 - q[2] **2) - accelerometer[2],
            2*b[1]* (.5 - q[2] **2 - q[3] **2) + 2*b[3]* (q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
            2*b[1]* (q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
            2*b[1]* (q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - magnetometer[2],
        ]        
        j = [
            [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0],
            [-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]],
        ]
        f = [sensor_fusion] # vector->matrix
        step = self._dot(self._transpose(j),f)
        step = step[0] #matrix -> vector
        step = self._div(step, self._norm(step))
        qdot = self._q_mul_q(q, [0.] + gyroscope)
        qdot = self._div(qdot, 2)
        qdot = self._q_minus_q(qdot , step)
        qdot = self._div(qdot, 1/self.dt)
        q = self._q_add_q(q,qdot)
        q = self._div(q, self._norm(q))
        self.quaternion = q

    # MATH METHODS
    def _differentiate(self,y2,y1):
        return (y2-y1)/self.dt

    def _integrate(self,a,b):
        # integration is equivalent to finding the area under the curve 
        # area under 2 points (i.e. assumed to be a straight line) can be approximated to the area of a trapezium h*(a+b)/2
        return .5 * self.dt * (a+b)

    def _minima(self):
        # finds the minimum magnitude of acceleration in a given window (e.g. 200 prev steps)
        return self.magnitudes.index(min(self.magnitudes))

    def _drift(self,velocity,localMinima):
        # the most recent time point where the magnitude of all acceleration components was minimal (the local minima)
        # is assumed to be where the device is not moving (at rest)
        # with this reference point we can zero the velocity at that exact point
        # and use it to approximate the drift in velocity
        # corrected velocity returned
        drift = [abs(v) for v in self.velocities_drifted[localMinima]]
        return [v+d if v < 0 else v-d for v,d in zip(velocity,drift)]      
        
    # MATRIX METHODS    
    def _transpose(self,matrix): 
        return [[row[col] for row in matrix] for col in range(len(matrix[0]))]
    
    def _dot(self,matrix1,matrix2):
        return self._transpose([[sum(a*b for a,b in zip(A,B)) for B in matrix2] for A in matrix1])

    def _inverse(self,matrix):
        new_matrix = [row + row[:2] for row in matrix]
        diag1,diag2 = 0,0
        for col in range(3):
            x,y= 1,1
            for row in range(3):
                x *= new_matrix[row][row+col]
                y *= new_matrix[row][4-row-col]
            diag1 += x
            diag2 += y
        determinant = diag1-diag2
        new_matrix.append(new_matrix[0])
        new_matrix.append(new_matrix[1])
        new_matrix = [row[1:] for row in new_matrix][1:]
        return [[(r2[c2]*r1[c1]-r2[c1]*r1[c2])/determinant for r1,r2 in zip(new_matrix,new_matrix[1:]) ] for c1,c2 in zip(range(3),range(1,4)) ]

    # VECTOR METHODS
    def _div(self,vec,n):
        return [v/n for v in vec]

    def _norm(self,vec):
        #the sqrt (of the sum (of all values squared))
        return sum([v**2 for v in vec]) ** .5

    # QUATERNION METHODS
    def _conj(self,quaternion):
        #all values but the first element in quaternion are inverted
        return [-q if i > 0 else q for i,q in enumerate(quaternion)]

    def _q_mul_q(self,quaternion1, quaternion2):
        #multiply two quaternions together
        a,b,c,d = quaternion1
        aa,bb,cc,dd = quaternion2
        w = a*aa - b*bb - c*cc - d*dd
        x = a*bb + b*aa + c*dd - d*cc
        y = a*cc - b*dd + c*aa + d*bb
        z = a*dd + b*cc - c*bb + d*aa
        return [w,x,y,z]

    def _q_add_q(self,quaternion1,quaternion2):
        # add two quaternions together element-wise
        return [q1 + q2 for q1,q2 in zip(quaternion1,quaternion2)]
    
    def _q_minus_q(self,quaternion1,quaternion2):
        # subtract two quaternions together element-wise
        return [q1 - q2 for q1,q2 in zip(quaternion1,quaternion2)]

    def _matrix(self,quaternion):
        #converts a quaternion into a 3x3 rotation matrix
        w,x,y,z = quaternion

        t0 = 1 - 2* (y**2 - z**2)
        t1 = 2* (x*y + w*z) 
        t2 = 2* (x*z - w*y) 
        t3 = 2* (x*y - w*z)
        t4 = 1 - 2* (x**2 - z**2)
        t5 = 2* (y*z + w*x) 
        t6 = 2* (x*z + w*y)
        t7 = 2* (y*z - w*x)
        t8 = 1 - 2* (x**2 - y**2) 

        return [[t0,t1,t2],[t3,t4,t5],[t6,t7,t8]]
        
    def _angles(self,quaternion):
        # converts a quaternion into euler angles (in degrees NOT rads)
        # Roll = X-axis rotation
        # Pitch = Y-axis rotation
        # Yaw = Z-axis rotation
        from math import atan2,asin
        PI = 3.14159265359
        RAD2DEG = 180/PI
        w,x,y,z = quaternion

        sinX = 2* (w*x + y*z)
        cosX = 1 - 2*(x**2 + y**2)
        roll = atan2(sinX,cosX)

        sinY = max(-1., min(1., 2* (w*y - z*x)))
        pitch = asin(sinY)

        sinZ = 2* (w*z + x*y)
        cosZ = 1 - 2* (y**2 + z**2)
        yaw = atan2(sinZ,cosZ)

        return [roll*RAD2DEG, pitch*RAD2DEG, yaw*RAD2DEG]

    '''
    def _gravity(self,acceleration,quaternion):
        # return vector represnting gravity's influence on device
        w,x,y,z = quaternion
        return [2* (x*z - w*y), 2* (w*x + y*z), w**2 - x**2 - y**2 + z**2]
    '''

odometer = IMU()
for _ in range(500):
    degrees_freedom = odometer.step()
    
