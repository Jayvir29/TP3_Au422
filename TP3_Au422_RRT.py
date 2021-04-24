#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Imports

from __future__ import print_function
import rospy
import cv2
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.srv import GetMap
from nav_msgs.msg import Path
from nav_msgs.msg import MapMetaData
from sys import exit
import numpy as np
import matplotlib.pyplot as plt
import copy

class RRT:
    def __init__(self, K=0, dq=0):
        
        ## Variable de classe ##
        
        # Coordonnées du point de départ 
        self.x_init = 0.0 #init x 
        self.y_init = 0.0 #init y
        self.z_init = 0.0 #init z

        # Coordonnées du point d'arrivée 
        self.x_goal = 0.0 # goal position x
        self.y_goal = 0.0 # goal position y
        self.z_goal = 0.0 # goal position z

        # Variable associé à la map (fixe)
        self.x_map_origin = 0.0 # origin position x
        self.y_map_origin = 0.0 # origin position y
        self.map_resolution = 0.0 # resolution of the map
        self.map_width = 0.0 # width of the map
        self.map_height = 0.0 # height of the map

        # Variable associés à RRT
        self.K = K # Number of iteration of the tree
        self.dq = dq # distance between each node

        """ Constructor """
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.starting_pose_cb) #Get the initial position
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_parameters) #Get the parameters of the map
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_cb) #Get the goal position
        self.pathPub = rospy.Publisher("/path", Path, queue_size=1) #Publish the path

        print("Waiting for map service to be available...\n")
        rospy.wait_for_service('/static_map')   #The node waits for the service that provides the map to be avaiblable
        try:    #GET the map data
            get_map = rospy.ServiceProxy('/static_map', GetMap)
            self.map = get_map().map
            print("Map received !\n")
            print("Please, select your starting and ending point.\n")
            print ("-----------------------------------------------------------------------------")
            

        except rospy.ServiceException as e:
            print("Map service call failed: %s"%e)
            exit()

        """ TODO - Add your attributes """
        self.starting_pose_received = False

    # **********************************
    def starting_pose_cb(self, msg): #Get the coordinates of the starting point in Image frame
        """ STUDENT TODO - Get the starting pose """

        self.x_init = int((msg.pose.pose.position.x - self.map.info.origin.position.x)/self.map.info.resolution)
        self.y_init = self.map.info.height - int((msg.pose.pose.position.y - self.map.info.origin.position.y)/self.map.info.resolution)
        print("Coordinates of starting point : \n",self.x_init,self.y_init)
        print ("-----------------------------------------------------------------------------")

        self.starting_pose_received = True

    # **********************************
    def goal_pose_cb(self, msg): #Get the coordinates of the ending point in Image frame
        """ TODO - Get the goal pose """
    
        self.x_goal = int((msg.pose.position.x - self.map.info.origin.position.x)/self.map.info.resolution)
        self.y_goal = self.map.info.height - int((msg.pose.position.y - self.map.info.origin.position.y)/self.map.info.resolution)
        print("Coordinate of ending point : \n",self.x_goal,self.y_goal)
        print ("-----------------------------------------------------------------------------")

        #TO DOT TOUCH
        if self.starting_pose_received:
            self.run()

    # **********************************
    def map_parameters(self, msg): #Get the map parameters

        self.x_map_origin = msg.origin.position.x
        self.y_map_origin = msg.origin.position.y
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        print ("Metada map :\n")
        print("Map origin x : ",self.x_map_origin,"\n Map origin y : ",self.y_map_origin,"\n Map width : ",self.map_width,"\n Map height : ",self.map_height,"\n Map resolution : ",self.map_resolution)
        print ("-----------------------------------------------------------------------------")

    # **********************************

    def run(self): #Main function
        """ TODO - Implement the RRT algorithm """

        qnear = np.array([self.x_init, self.y_init])
        print ("\n Initial qnear : \n",qnear)
        self.vertex = np.array([qnear])
        print ("\n Initial vertex : \n",self.vertex)
        self.edge = np.array([ qnear, qnear ])
        print ("\n Initial edge : \n",self.edge)
        print ("-----------------------------------------------------------------------------")

        qnew = qnear #Initially
        

        for i in range (self.K):

            dist = self.calc_distance(np.array([self.x_goal,self.y_goal]),qnew)
            if dist < self.dq : #If the distance between qnew and the goal node is less than delta q
                qgoal = np.array([self.x_goal,self.y_goal])
                self.add_vertex(qgoal)
                self.add_edge(qnew,qgoal)
                print ("Target achieved !")
                break

            else : #If the distance between qnew and the goal node is higher than delta q
                print ("**************************************")
                print ("Interation n° :",i)
                qrand = self.Random_free_configuration()
                print ("\n Generation of qrand : \n",qrand)
                qnear = self.Nearest_vertex(qrand)
                print ("\n Generation of qnear : \n",qnear) # Keep in mind that the first qnear is also the first node so qinit, which is here not represented at the beginning
                qnew = self.New_conf(qnear,qrand)
                print ("\n Generation of qnew : \n",qnew)
                b = self.free_or_not(qnew)
                if b == True : #If the cell is available
                    self.add_vertex(qnew)
                    self.add_edge(qnear,qnew)
                else : #Otherwise
                    continue
        
        print ("-----------------------------------------------------------------------------")
        print ("\n Final vertex : \n",self.vertex)
        print ("\n Final edges : \n",self.edge)
        print ("-----------------------------------------------------------------------------")
        

        # Path finding algorithm
        
        self.q_init = np.array([self.x_init, self.y_init])
        self.q_final = np.array([self.x_goal, self.y_goal])

        q_temp = np.array([1000,1000]) # Random node for the path finding algorithme

        self.list_path = np.array([ self.q_final ])
        
        while ( (q_temp[0] != self.q_init[0]) and (q_temp[1] != self.q_init[1]))  : #Block code to find the parent node from a its child
            
            for i in range (0,len(self.edge),2): #Only parent nodes
                if ( (self.edge[i][0] ==  self.list_path[-1][0] ) and (self.edge[i][1] ==  self.list_path[-1][1] ) ):
                    self.list_path = np.vstack((self.list_path, np.array([ self.edge[i+1] ])))
                    q_temp = self.edge[i+1]
            print ("Currently working on parents node...\n(NB: If this is displayed infinitely, then the parent node is missing)")
    
        print ("Original path found : \n",self.list_path)
    
        # Reduction of the path

        self.list_path_apres_reduction = self.reduce_path(self.list_path) #First reduction of the path


        self.list_path_apres_reduction_temp = [] 

        if (len(self.list_path_apres_reduction) > 2 ): #Block code : Reduce the path again, if the path contains more than 2 nodes
            self.list_path_apres_reduction_temp.append(self.list_path_apres_reduction[-1])
            print ("Element pop ! ",self.list_path_apres_reduction.pop(-1))
            self.list_path_apres_reduction = self.reduce_path(self.list_path_apres_reduction)
            self.list_path_apres_reduction.append(self.list_path_apres_reduction_temp[0])
            

        # Convertion of the 2D-list to a 2D-array (for our purposes)

        self.list_path_reduced = np.array([ self.q_final ])
        for reduced_point in self.list_path_apres_reduction :
            self.list_path_reduced = np.vstack((self.list_path_reduced, np.array([ reduced_point[0] , reduced_point[1] ])))

        print ("Final reduced path : \n",self.list_path_reduced)

        self.publishPath()
        

        # Display the original and reduced path in a matplotlib.pyplot figure

        plt.figure()
        for i in range (len(self.vertex)):
            plt.plot(self.vertex[i][0],self.vertex[i][1],'bx',label="Nodes")
        plt.xlim(0,self.map.info.width)
        plt.ylim(0,self.map.info.height)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.plot(self.list_path[:,0],self.list_path[:,1],'g-',label="Original Path")
        plt.plot(self.x_init,self.y_init,'ro', label="Start & End points")
        plt.plot(self.x_goal,self.y_goal,'ro')
        plt.plot(self.list_path_reduced[:,0],self.list_path_reduced[:,1],'m-',label="Reduced Path")
        plt.legend()
        plt.title("Implementation of the original and reduced path, computed with RRT Algorithm")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # Display the path in the map

        self.display_map()


        # Display the path in the Image


        self.img = copy.deepcopy(self.occupancy_array) #Get the occupancy_array to another variable which will be modified
        
        #To keep the occupancy_array unchanged for computations

        blue = (255, 0, 0) #Red color in BGR
        red = (0, 0, 255) #Blue color in BGR
        green = (0,255,0) #Green color in BGR
        magenta = (255,0,255) #Magenta color in BGR
        thickness_path = 10
        thickness_vertex = 5
        thickness_points = 15

        for i in range(len(self.vertex)): #Display all nodes
            self.img = cv2.circle(self.img, ( int(self.vertex[i][0]), int(self.vertex[i][1]) ), 1, blue, thickness_vertex)

        for i in range(len(self.list_path)-1): #Display original path
            self.img = cv2.line(self.img, ( int(self.list_path[i][0]), int(self.list_path[i][1]) ), ( int(self.list_path[i+1][0]), int(self.list_path[i+1][1]) ), green, thickness_path)
        
        for i in range(len(self.list_path_reduced)-1): #Display reduced path
            self.img = cv2.line(self.img, ( int(self.list_path_reduced[i][0]), int(self.list_path_reduced[i][1]) ), ( int(self.list_path_reduced[i+1][0]), int(self.list_path_reduced[i+1][1]) ), magenta, thickness_path)
        
        #Display starting and ending points

        self.img = cv2.circle(self.img, ( self.x_init, self.y_init ), 1, red, thickness_points)
        self.img = cv2.circle(self.img, ( self.x_goal, self.y_goal ), 1, red, thickness_points)

        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyWindow('image')
        
        pass

    # **********************************
    def publishPath(self):
        """ Send the computed path so that RVIZ displays it """
        """ TODO - Transform the waypoints from pixels coordinates to meters in the map frame """
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        path_RVIZ = []
        for pose_img in self.list_path_reduced :
            pose = PoseStamped()
            pose.pose.position.x = ( (pose_img[0] * self.map_resolution) - (  (self.map_width/2) * self.map_resolution ) )
            pose.pose.position.y = (  (  (self.map_height/2) * self.map_resolution )  - (pose_img[1] * self.map_resolution)  )
            path_RVIZ.append(pose)
        print ("type path_RVIZ complete !",type(path_RVIZ))
        msg.poses = path_RVIZ
        self.pathPub.publish(msg)
    
    # **********************************
    def display_map(self): # Method to display the image of the map using the data of the map itself

        self.occupancy_array = np.zeros((self.map.info.height,self.map.info.width,3))
        for i in range(self.map.info.height):
            for j in range(self.map.info.width):
                if (self.map.data[i*self.map.info.width+j] == 0):
                    self.occupancy_array[i,j,:] = 255
        self.occupancy_array = np.flipud(self.occupancy_array)

    # **********************************
    def Random_free_configuration(self): # Generate qrand
        x = int(np.random.randint(self.map.info.width, size=1))
        y = int(np.random.randint(self.map.info.height, size=1))
        return np.array([x, y])

    # **********************************
    def Nearest_vertex(self,qrand): # Find qnear
        n = len(self.vertex)
        distances = np.array([])
        qnear = np.array([])
        for i in range(n):
            calcul_distance = np.sqrt(pow((qrand[0] - self.vertex[i][0]),2) + pow(qrand[1] - self.vertex[i][1],2)) # Compute the distance between qrand and all the founded nodes
            distances = np.append(distances, calcul_distance)
        argmin_ = np.argmin(distances) #We want the lower distance
        qnear = self.vertex[argmin_]
        return qnear

    # **********************************
    def New_conf(self,qnear,qrand): # Find qnew
        arg = np.arctan2((qrand[1] - qnear[1]),(qrand[0] - qnear[0])) #Arg for the direction
        qnew = np.array([qnear[0] + (self.dq * np.cos(arg)), qnear[1] + (self.dq * np.sin(arg))])
        return qnew

    # **********************************
    def free_or_not(self, qnew): # Check if the cells are available (= without obstacle) or not

        #Get the data of the map in a matrix
        occupancy = self.map.data
        self.occupancy_array = np.array(occupancy)
        self.occupancy_array = self.occupancy_array.reshape((self.map.info.height, self.map.info.width))
        self.occupancy_array = np.flipud(self.occupancy_array)

        x_temp = int(qnew[0])
        y_temp = int(qnew[1])

        if (y_temp >= self.map.info.height) or (x_temp >= self.map.info.width): # If x_temp and y_temps are out of axis
            print("Out of map\n")
            b = False
        else : 
            if (self.occupancy_array[y_temp,x_temp] == 0) : #If the cell is available
                b = self.check_obstacle(self.vertex[-1],qnew) #Check if there is an obstacle in the path between qnew and the last computed node

            else: #Otherwise
                b = False
        
        return b

    # **********************************
    def add_vertex(self, qnew): # Append new nodes in self.vertex
        self.vertex = np.append(self.vertex, [qnew], axis=0)

    # **********************************
    def add_edge(self, qnew, qnear): # Append new pairs of child-parent nodes in self.edge
        
        edge = np.array([ qnear, qnew ])
        self.edge = np.vstack((self.edge, edge))

    # **********************************
    def calc_distance(self,qrand,qnear): #Compute a distance between 2 nodes
        return (np.sqrt(pow((qrand[0] - qnear[0]),2) + pow(qrand[1] - qnear[1],2)))

    # **********************************
    def check_obstacle(self,q1,q2): # Check if there is an obstacle between 2 nodes by discretizing the distance between them 
                                    # And test every single intermediate node created by the discretization
    
        liste_verification = []
        distance_q1_q2 = np.sqrt(pow((q2[1] - q1[1]),2) + pow((q2[0] - q1[0]),2))
        
        for i in np.arange (0, distance_q1_q2 , distance_q1_q2/50)  : #Block code : Discretization
            
            arg = np.arctan2((q2[1] - q1[1]),(q2[0] - q1[0]))
            qintermediaire = np.array([q1[0] + (  i * np.cos(arg)), q1[1] + ( i  * np.sin(arg))])

            if (self.occupancy_array[ int(qintermediaire[1])][ int(qintermediaire[0]) ] == 0) : #Block code : check if each intermediate node is available
                r1 = True
                liste_verification.append(r1)
        
            else:
                r1 = False
                liste_verification.append(r1)
            
        for i in liste_verification : #Block code : If all intermediate nodes are available, then the 2 inputted nodes (q1 and q2) can be linked
            if i == False:
                r2 = False # return false if obstacle
                break   #Break the code if at least one intermediate node is at an obstacle
            else : 
                r2 = True # return true if not obstacle
        
        return r2

    # **********************************
    def reduce_path(self,list_path): #Function to reduce the path
    
        qi = [list_path[-1][0],list_path[-1][1]] #Start from the end of the original path
        
        # Transform the array which contains the path to a list
        lp = []
        for i in range (len(list_path)):
            lp.append([list_path[i][0],list_path[i][1]]) 
        
        # Creation of the new path reduced 
        list_path_reduced = []

        working_node = len(lp)-1
        
        j = 0 # Security Variable to avoid infinite loop

        while (qi not in list_path_reduced) : # Waiting for the initial node to be in the reduced path
            i = 0
            add_or_not = False
            #print ("Error : First node is not in the reduced path\n")

            j = j + 1
            if j == 15 : # No more than 15 loops
                break

            while (not (add_or_not)) and (i < working_node) : # We studied the distance of the i and working_node element of the list path
            
                r = self.check_obstacle(lp[working_node], lp[i])

                if r == True :
                    #print ("Error : New node is added infinitely")
                    add_or_not = True
                    list_path_reduced.append(lp[i]) # The 2 nodes do not require any intermediate node to be linked 
                    list_path_reduced.append(lp[working_node])  

                    working_node = i
                    # i = len(list_path_reduced)

                else:  # The 2 nodes require intermediate nodes to be linked
                    list_path_reduced.append(lp[i])
                    i+=1

        if qi not in list_path_reduced : # Safety for the first node
            list_path_reduced.append(qi)

        print ("Nouvelle reduction !")
        return list_path_reduced


if __name__ == '__main__':
    # roslaunch au422_tp3 my_rviz_simu.launch
    # rosrun au422_tp3 path_planning_rrt.py

    """ DO NOT TOUCH """
    rospy.init_node("RRT", anonymous=True)

    rrt = RRT(2000,100) # (K=iterations ,dq= distance between each nodes (step))
    rospy.spin()