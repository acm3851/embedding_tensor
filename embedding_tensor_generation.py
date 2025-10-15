from matplotlib import pyplot as plt
import numpy as np
import scipy
import pickle
import time
#import threading
#import concurrent.futures
import yaml
#import openmpi
from scipy import sparse

#from matplotlib.pyplot import figure

def generate_init_tensor(num_polygons):
    #we are doing inheritance-based search, so set the number of polygons to 1.
    #num_polygons = 1
    
    #we are assuming here that everything is 3-connected.
    #the 3-connectedness assists us with creating the hexagonal grid.
    #If you need different number of connections,
    #you would need a different setup.

    #now we need to embed the atoms within a unit cell.
    #let's make this cell a honeycomb
    #since the number of polygons could be prime,
    #let's just stack hexagons vertically.
    v_num = 4*2*num_polygons
    e_num = 4*3*num_polygons
    f_num = 4*num_polygons
    embedding_tensor = np.zeros((v_num,e_num,f_num))#vertices, edges, faces
    # we end up with two sets of vertices, a left half and a right half.
    # we multiply by 4 to ensure that the tensor does not have face self-intersections

    #now we write everything into the tensor
    #(see diagram)
    #edge class 6n has vertices (4n+3,4n) and faces (2n,2n+2)
    #edge class 6n+1 has vertices (4n,4n+1) and faces (2n+2,2n+1)
    #edge class 6n+2 has vertices (4n+1,4n+4) and faces (2n+2,2n+3)
    #edge class 6n+3 has vertices (4n+1,4n+2) and faces (2n+1,2n+3)
    #edge class 6n+4 has vertices (4n+2,4n+3) and faces (2n+3,2n)
    #edge class 6n+5 has vertices (4n+3,4n+6) and faces (2n+2,2n+3)
    #this creates the honeycomb.
    V_array = [(3,0),(0,1),(1,4),(1,2),(2,3),(3,6)]
    F_array = [(0,2),(2,1),(2,3),(1,3),(3,0),(2,3)]
    for n in range(2*num_polygons):
        for edge_set in range(6):
            embedding_tensor[(4*n+V_array[edge_set][0])%v_num,(6*n+edge_set)%e_num,(2*n+F_array[edge_set][0])%f_num]=1
            embedding_tensor[(4*n+V_array[edge_set][0])%v_num,(6*n+edge_set)%e_num,(2*n+F_array[edge_set][1])%f_num]=1
            embedding_tensor[(4*n+V_array[edge_set][1])%v_num,(6*n+edge_set)%e_num,(2*n+F_array[edge_set][0])%f_num]=1
            embedding_tensor[(4*n+V_array[edge_set][1])%v_num,(6*n+edge_set)%e_num,(2*n+F_array[edge_set][1])%f_num]=1
            
    #the hexagonal tiling should be made already
    #with the following
    #face maps (horizontal) (2n -> 2n+1)
    #face maps (vertical) (2n -> 2n + poly)
    #edge maps (horizontal) with (4n -> 4n+2), (4n+1 -> 4n+3)
    #edge maps (vertical) with (4n -> 4n+(2*poly)), (4n+1 -> 4n+1+(2*poly))
    #vertex maps (horizontal) with (6n -> 6n+3), (6n+1 -> 6n+4), (6n+2 -> 6n+5)
    #vertex maps (vertical) with (6n -> 6n+(3*poly)), (6n+1 -> 6n+1+(3*poly)), (6n+2 -> 6n+2+(3*poly))
    return embedding_tensor

def tensor_slice(embedding_tensor,graph_detail,face_to_split,edge_1,edge_2,vertex):
    
    pos_init, X_coord_adj, Y_coord_adj = graph_detail
    #adding one face, 3 edges, 2 vertices
    #we split face_to_split in 2, connecting edge_1 and edge_2
    #0,1,2

    #adding pos_init as a hack to get an approximation of the atomic positions
    
    #sanity check
    #have to check if both edges are *in* the face.
    if np.sum(embedding_tensor[:,edge_1,face_to_split])==0 or np.sum(embedding_tensor[:,edge_2,face_to_split])==0:
        print("edges do not both belong in face")
        raise Exception
    if edge_1==edge_2:
        print("edges are identical")
        raise Exception
    #now the hard part

    #what are the faces that border edge_1 and edge_2 that are not face_to_split?
    face_1 = list(set(np.where(np.sum(embedding_tensor[:,edge_1,:],axis=0)==2)[0])-set([face_to_split]))[0]
    face_2 = list(set(np.where(np.sum(embedding_tensor[:,edge_2,:],axis=0)==2)[0])-set([face_to_split]))[0]
    #these are face_1 for edge_1 and face_2 for edge_2
    
    f0,f1,f2=embedding_tensor.shape
    new_tensor = np.zeros((2+f0,3+f1,1+f2))
    new_tensor[:f0,:f1,:f2]=embedding_tensor
    #embed the old into the new
    
    #let's grab the vertices involved in the edges
    #edge identity stays with the lowest-valued vertex
    
    v=np.sum(new_tensor[:,edge_1,:],axis=1)#sum over axis=1 to get rid of face-dependency.
    
    v1set = np.where(v==2)[0]
    if not(vertex in v1set):
        print("vertex not in edge 1")
        raise Exception
    b = vertex
    a = list(set(v1set)-set([b]))[0]
    
    v=np.sum(new_tensor[:,edge_2,:],axis=1)#sum over axis=1 to get rid of face-dependency.
    c,d = np.where(v==2)[0]
    
    #the resulting vertices of the face need to be split between it and the new one, somehow.
    face_array = embedding_tensor[:,:,face_to_split]
    #we want to label the vertices *in order* that we want to cut out.

    #let's just start this with vertex "a"
    vertices_to_relabel=[]
    vertex_prev = b
    
    #start_edges = set(np.where(face_array[vertex_start]==1)[0])
    #travel_edge = start_edges - set([edge_1,edge_2])
    travel_edge = set([edge_1])
    
    #the edges are cut in two, and we want to travel the other way,
    #labelling all the vertices we have to relabel on the way.

    xyb_travel = [0,0]#what are the lines I cross from b to c,d?
    while True:        
        next_vertex = list(set(np.where(face_array[:,list(travel_edge)[0]])[0])-set([vertex_prev]))[0]

        xyb_travel[0]+=X_coord_adj[vertex_prev][next_vertex]
        xyb_travel[1]+=Y_coord_adj[vertex_prev][next_vertex]

        vertices_to_relabel.append(next_vertex)
        next_edges = set(np.where(face_array[next_vertex]==1)[0])
        
        #if len(next_edges)==3:
        #    print(face_array[next_vertex])

            
        travel_edge = next_edges - travel_edge
        vertex_prev = int(next_vertex)
        #loop now working
        #next, we stop when we hit the other one
        if list(travel_edge)[0] in [edge_1,edge_2]:
            break
    
    #take the vertices we relabeled and put them in the new face.
    #and remove the references in the old face.
    new_tensor[vertices_to_relabel,:,-1] = new_tensor[vertices_to_relabel,:,face_to_split]
    new_tensor[vertices_to_relabel,:,face_to_split]*=0

    #and we have to break the edges in half now, too.
    #fortunately, it is far easier.
    new_tensor[:,edge_1,:]*=0
    new_tensor[:,edge_2,:]*=0
    
    #replaced with
    #(a,edge_1,face_1), (a,edge_1,-1)
    #(b,-1,face_1), (b,-1,face_to_split)
    #(-1,edge_1,face_1),(-1,edge_1,-1),(-1,-1,face_1),(-1,-1,face_to_split),(-1,-3,-1),(-1,-3,face_to_split)
    #and the next is dependent on if we hit "c" or "d".

    new_tensor[a,edge_1,face_1]=1
    new_tensor[a,edge_1,-1]=1
    new_tensor[b,-1,face_1]=1
    new_tensor[b,-1,face_to_split]=1
    #fixing the connections of a and b
     
    new_tensor[-1,edge_1,face_1]=1
    new_tensor[-1,edge_1,-1]=1
    new_tensor[-1,-1,face_1]=1
    new_tensor[-1,-1,face_to_split]=1
    new_tensor[-1,-3,-1]=1
    new_tensor[-1,-3,face_to_split]=1
    #creating the connections of a new vertex "-1"
    #together, they fix edges "edge_1" and "-1"
    
    X2 = np.zeros((2+f0,2+f0))
    X2[:f0,:f0] = np.array(X_coord_adj)

    Y2 = np.zeros((2+f0,2+f0))
    Y2[:f0,:f0] = np.array(Y_coord_adj)

    X2[a][b]=0
    X2[b][a]=0
    Y2[a][b]=0
    Y2[b][a]=0
    X2[c][d]=0
    X2[d][c]=0
    Y2[c][d]=0
    Y2[d][c]=0

    #print(np.sum(new_tensor,axis=1))
    #print(np.sum(np.sum(new_tensor,axis=2),axis=0))
    pos2 = np.zeros((new_tensor.shape[0],2))
    pos2[:-2] = np.array(pos_init)
    #print(a,b,c,d)
    #print(X_coord_adj[a][b])
    
    pos2[-2] = (pos2[c]+pos2[d]+np.array([X_coord_adj[c][d],Y_coord_adj[c][d]]))/2

    pos2[-1] = (pos2[a]+pos2[b]+np.array([X_coord_adj[a][b],Y_coord_adj[a][b]]))/2
    

    #settling -1,a,b
    da = abs(pos2[-1]-pos2[a])
    if 0 < da[0] < 1:
        #-1 is close to a
        #X[a][-1]=0
        X2[b][-1]=X_coord_adj[b][a]
        X2[-1][b]=-X2[b][-1]
    else:
        #-1 is close to b
        X2[a][-1]=X_coord_adj[a][b]
        X2[-1][a]=-X2[a][-1]
    
    if 0 < da[1] < 1:
        #-1 is close to a
        Y2[b][-1]=Y_coord_adj[b][a]
        Y2[-1][b]=-Y2[b][-1]
    else:
        #-1 is close to b
        Y2[a][-1]=Y_coord_adj[a][b]
        Y2[-1][a]=-Y2[a][-1]
    #'''
    #settling -2,c,d
    dc = abs(pos2[-2]-pos2[c])
    if 0 < dc[0] < 1:
        #-2 is close to c
        #X[c][-2]=0
        X2[d][-2]=X_coord_adj[d][c]
        X2[-2][d]=-X2[d][-2]
    else:
        #-2 is close to d
        X2[c][-2]=X_coord_adj[c][d]
        X2[-2][c]=-X2[c][-2]
    
    if 0 < dc[1] < 1:
        #-2 is close to c
        Y2[d][-2]=Y_coord_adj[d][c]
        Y2[-2][d]=-Y2[d][-2]
    else:
        #-2 is close to d
        Y2[c][-2]=Y_coord_adj[c][d]
        Y2[-2][c]=-Y2[c][-2]
    #'''
    

    
    #d, -2
    #print(pos2[-2])
    
    if c in vertices_to_relabel:
        #print(X2[-1][b],Y2[-1][b])
        #print(xyb_travel)
        #print(X2[c][-2],Y2[c][-2])
        X2[-1][-2] = X2[-1][b] + xyb_travel[0] + X2[c][-2]
        X2[-2][-1] = - X2[-1][-2]

        Y2[-1][-2] = Y2[-1][b] + xyb_travel[1] + Y2[c][-2]
        Y2[-2][-1] = - Y2[-1][-2]

        '''
        X2[-2][-1] = X2[-2][c] + xyb_travel[0] + X2[b][-1]
        X2[-2][-1] = -X2[-1][-2]

        Y2[-2][-1] = Y2[-2][c] + xyb_travel[1] + Y2[b][-1]
        Y2[-2][-1] = -Y2[-1][-2]
        #'''

        #from a to c
        #(c,edge_2,face_2), (c,edge_2,-1)
        #(d,-2,face_2), (d,-2,face_to_split)
        #(-2,edge_2,face_2),(-2,edge_2,-1),(-2,-2,face_2),(-2,-2,face_to_split),(-2,-3,-1),(-2,-3,face_to_split)
        new_tensor[c,edge_2,face_2]=1
        new_tensor[c,edge_2,-1]=1
        new_tensor[d,-2,face_2]=1
        new_tensor[d,-2,face_to_split]=1
    else:
        #from a to d
        #same, but c and d are swapped.

        X2[-1][-2] = X2[-1][b] + xyb_travel[0] + X2[d][-2]
        X2[-2][-1] = - X2[-1][-2]

        Y2[-1][-2] = Y2[-1][b] + xyb_travel[1] + Y2[d][-2]
        Y2[-2][-1] = - Y2[-1][-2]

        new_tensor[d,edge_2,face_2]=1
        new_tensor[d,edge_2,-1]=1
        new_tensor[c,-2,face_2]=1
        new_tensor[c,-2,face_to_split]=1

    new_tensor[-2,edge_2,face_2]=1
    new_tensor[-2,edge_2,-1]=1
    new_tensor[-2,-2,face_2]=1
    new_tensor[-2,-2,face_to_split]=1
    new_tensor[-2,-3,-1]=1
    new_tensor[-2,-3,face_to_split]=1
    #to fix the antisymmetry, we have to swap the labels of edge_2 and -2
    #print(np.sum(new_tensor,axis=1))

    #normal swap operations don't work - this does though
    temp = np.array(new_tensor[:,edge_2,:])
    new_tensor[:,edge_2,:] = np.array(new_tensor[:,-2,:])
    new_tensor[:,-2,:] = np.array(temp)
    
    graph_detail_return = (pos2, X2, Y2)
    return new_tensor,graph_detail_return

def generate_graph_details():
    #assumes n=1
    Z=np.zeros((8,2))
    Z[0]=[1/6,1/6]
    Z[1]=[2/6,2/6]
    Z[0:2]+=[0.2,-0.2]
    Z[2:4]=Z[0:2]+[0.5,0]
    Z[4:6]=Z[0:2]+[0,0.5]
    Z[6:8]=Z[0:2]+[0.5,0.5]
    #Z is graph positions
    X_coord_adj=np.zeros((8,8))
    Y_coord_adj=np.zeros((8,8))

    Y_coord_adj[0][5]=-1
    Y_coord_adj[5][0]=1
    Y_coord_adj[2][7]=-1
    Y_coord_adj[7][2]=1

    X_coord_adj[3][0]=1
    X_coord_adj[0][3]=-1
    X_coord_adj[7][4]=1
    X_coord_adj[4][7]=-1
    return (Z,X_coord_adj,Y_coord_adj)
    
def double_flag_to_str(item0,item1):
    return tuple(item0)+tuple(item1)
    #return tuple(item0)

def double_bfs(tensor_0,tensor_1,flag_0,flag_1,unique_calc=False):
    unique_set = set()
    derivative_set = set()
    if tensor_0.shape != tensor_1.shape:
        return False,None
    flag_0 = np.array(flag_0)
    flag_1 = np.array(flag_1)
    #sanity check, see if the flags are in the tensors they claim to be
    if tensor_0[tuple(flag_0)]==0:
        print("flag_0 not in tensor_0")
        raise Exception
    if tensor_1[tuple(flag_1)]==0:
        print("flag_1 not in tensor_1")
        raise Exception
    if len(tensor_0.shape)!=len(tensor_1.shape):
        print("tensors do not match in dimension")
        raise Exception
    flag_maps=[dict(),dict(),dict()]
    sym_parity=False
    #sym_parity is the answer to the question:
    #does this symmetry (if it exists), mirror the plane?
    sym_n = 0
    double_sym_flag = []
    fix_items=[]
    #sym_n is the maximum number of fixed elements
    queue=[(flag_0,flag_1,sym_parity)]
    #running a breadth first search on both tensors simultaneously
    #looking for local inconsistency. (large-scale inconsistency ok)
    #in reference to the paper, this is the "lax" symmetry algorithm
    #stored in (flag 0, flag 1, parity)
    #the parity bit is meant to determine if the symmetry respects 
    visit_set=dict()
    while queue: 
        item0,item1,parity = queue.pop()
        if (item0==flag_1).all():
            double_sym_flag = item1
            sym_parity = not(parity)
        for i in range(len(item0)):#here is the issue - it never hits certain polygons.
            flag_maps[i][item0[i]]=item1[i]
            flag_maps[i][item1[i]]=item0[i]
        #print(item0,item1)
        #I calculate the number of fixed elements here, and update the max counter.
        fixed_items = np.sum(np.array(item0)==np.array(item1))
        if fixed_items > sym_n:
            sym_n = fixed_items
            fix_items = []
            fix_items.append((item0,item1))
        elif fixed_items == sym_n:
            fix_items.append((item0,item1))
        #fixed_items should now contain *all* such items.
        #if fixed_items==1:
        #    print(item0,item1)
        visit_set[double_flag_to_str(item0,item1)]=parity
        if unique_calc:
            #goal is to find a set of flags that form an orbifold, or the set of flags that are unique under all symmetry maps
            #so for every pair of flags, this set stores the first one
            #forming a set of flags that are unique under this specific symmetry
            p0 = tuple(item0) in unique_set
            p1 = tuple(item1) in unique_set
            p2 = tuple(item0) in derivative_set
            p3 = tuple(item1) in derivative_set
            if not(p0 or p1 or p2 or p3):
                #none found
                unique_set.add(tuple(item0))
                derivative_set.add(tuple(item1))
            else:
                #at least one found.
                if not(p0 or p2):
                    #item0 not found, item1 exists
                    derivative_set.add(tuple(item0))
                elif not(p1 or p3):
                    #item1 not found, item0 exists
                    derivative_set.add(tuple(item1))
                else:
                    if p0 and p1:
                        #what if both items exist?
                        derivative_set.add(tuple(item1))
                        unique_set.remove(tuple(item1))
                    #otherwise do nothing
                

        #now add all the items to be visited.
        for i in range(len(tensor_0.shape)):
            #let's generate the lists
            
            #'''
            list_template_0 = list(item0)
            list_template_1 = list(item1)
            
            list_template_0[i] = slice(None)
            list_template_1[i] = slice(None)
            
            tensor_to_search0 = tensor_0[tuple(list_template_0)]
            tensor_to_search1 = tensor_1[tuple(list_template_1)]

            element_0 = np.where(tensor_to_search0==1)[0]
            element_new0 = element_0[element_0!=item0[i]][0]

            element_1 = np.where(tensor_to_search1==1)[0]
            element_new1 = element_1[element_1!=item1[i]][0]

            item_0_prime = list(item0)
            item_1_prime = list(item1)
            
            item_0_prime[i] = element_new0
            item_1_prime[i] = element_new1

            #this accesses the same section as the flag in question
            #the idea is that the output of this is a 1D array,
            #which would allow us to get the identity of the next flags.

            #test to see if the tensors locally agree. (if they don't, there isn't a lax symmetry map)
            #agree in face#, vertex#, edge#, whatever
            #unfortunately, in the general case, this would be a recursive call of something like
            #if symmetries_with_tags(tensor_to_search0,tensor_to_search1)
            
            #i can cut down on possibilities - the flags involved need to be the same too
            #question is - what flags work here?
            
            #if double_bfs(tensor_0,tensor_1,flag_0,flag_1)
            #but in 2D, it's a little simpler

            access0 = list([slice(None),slice(None),slice(None)])
            access0[i] = element_new0
            
            access1 = list([slice(None),slice(None),slice(None)])
            access1[i] = element_new1

            tensor_check_0 = tensor_0[tuple(access0)]
            tensor_check_1 = tensor_1[tuple(access1)]
            #print(np.sum(tensor_check_0),np.sum(tensor_check_1))
            #tensor_0
            #tensor_1
            if np.sum(tensor_check_0)!=np.sum(tensor_check_1):
                return False,None
            
            str_item = double_flag_to_str(item_0_prime,item_1_prime)
            
            #has new item already been visited?
            if not(str_item in visit_set):
                #to test to see if the symmetry respects orientability
                # flag_0 maps to flag_1
                # After travelling, we end up with (flag_1,X) somewhere.
                # And its symmetry flag gives us the symmetry flags for the whole transformation
                #print(item1,flag_0)
                #print(item1==flag_0)
                #if tensor_0.shape[0]==16:
                #    print(item_1_prime)
                #if (item_1_prime==flag_0).all():
                #    #print(item_0_prime,item_1_prime)
                #    sym_parity = not(parity)
                visit_set[double_flag_to_str(item0,item1)]=not(parity)
                queue.append((item_0_prime,item_1_prime,not(parity)))
    return True,[sym_parity,sym_n,fix_items,double_sym_flag,flag_maps,unique_set]
    #flag_1 is in the form (vertex, edge, face)
    #double breadth-first search
    pass

def tensor_to_flag_matrix(tensor):
    face_num,edge_num, vertex_num = tensor.shape
    #FEV form
    sigma_F = np.zeros_like(tensor).astype(int)
    sigma_V = np.zeros_like(tensor).astype(int)
    sigma_E = np.zeros_like(tensor).astype(int)
    V=[]
    for i in range(edge_num):#iterating over edges
        a,b=np.where(tensor[:,i])
        #faces, vertices
        a = list(set(a))
        b = list(set(b))
        #note, I needed to put the edges first so that the formation of the list of lists array is linear and not quadratic runtime.
        #I would recommend changing it later.
        sigma_F[a[0]][i][b[0]]=a[1]
        sigma_F[a[0]][i][b[1]]=a[1]
        sigma_F[a[1]][i][b[0]]=a[0]
        sigma_F[a[1]][i][b[1]]=a[0]

        sigma_V[a[0]][i][b[0]]=b[1]
        sigma_V[a[1]][i][b[0]]=b[1]
        sigma_V[a[0]][i][b[1]]=b[0]
        sigma_V[a[1]][i][b[1]]=b[0]
    for i in range(face_num):#iterate over faces
        for j in range(vertex_num):#iterate over vertices
            edges=np.where(tensor[i,:,j])[0]
            if len(edges)>0:
                a,b = edges
                #print(a,b)
                sigma_E[i][a][j]=b
                sigma_E[i][b][j]=a
    return sigma_F,sigma_E,sigma_V


def symmetry_2(maps_0,maps_1,tensor_0,tensor_1,flag_0,flag_1,unique_calc=False):
    #replacement for double_bfs, hopefully.
    #ok, now with the "maps" in there, we walk across them like in the bfs case.
    
    #additional information we need:
    sym_n = 0 #sym_n is the maximum number of fixed elements
    flag_maps=[dict(),dict(),dict()]#the flag maps
    fix_items = []#items fixed under the transformation
    sym_parity=False
    double_sym_flag = []
    unique_set = None # I don't want to deal with this right now
    
    face_arr0 = list(np.sum(np.sum(tensor_0,axis=0),axis=0)/2)
    face_arr1 = list(np.sum(np.sum(tensor_1,axis=0),axis=0)/2)
    face_arr0 = list(map(int,face_arr0))
    face_arr1 = list(map(int,face_arr1))
    
    search_item = tuple(flag_0)+tuple(flag_1)+(0,) #flag 0, flag 1, and the parity bit.
    queue = [search_item]
    visit_set=set()
    while queue:
        search_item = queue.pop()
        visit_set.add(search_item)
        #EV,F
        #EF,V
        #FV,E
        f0 = search_item[0]
        f1 = search_item[3]
        e0 = search_item[1]
        e1 = search_item[4]
        v0 = search_item[2]
        v1 = search_item[5]
        parity = search_item[6]

        if face_arr0[v0]!=face_arr1[v1]:
            return False, None

        item0 = np.array([f0,e0,v0])
        item1 = np.array([f1,e1,v1])
    
        if (item0 == flag_1).all():
            double_sym_flag = item1
            sym_parity = not(parity)#same as 1-, but converts to bool.
        
        #flag_maps computation
        #bidirectional? it shouldn't be?
        flag_maps[0][f0]=f1
        flag_maps[0][f1]=f0
        flag_maps[1][e0]=e1
        flag_maps[1][e1]=e0
        flag_maps[2][v0]=v1
        flag_maps[2][v1]=v0
        #weird...

        #calculating the maximum number of fixed items
        fix_item_num = (f0==f1) + (e0==e1) + (v0==v1)
        if fix_item_num > sym_n:
            sym_n = fix_item_num
            fix_items = []
            fix_items.append((item0,item1))
        elif fix_item_num == sym_n:
            fix_items.append((item0,item1))

        f0_prime = maps_0[0][f0][e0][v0]
        f1_prime = maps_1[0][f1][e1][v1]
        e0_prime = maps_0[1][f0][e0][v0]
        e1_prime = maps_1[1][f1][e1][v1]
        v0_prime = maps_0[2][f0][e0][v0]
        v1_prime = maps_1[2][f1][e1][v1]
        Fprime_flag = tuple([f0_prime,e0,v0,f1_prime,e1,v1,1-parity])
        Eprime_flag = tuple([f0,e0_prime,v0,f1,e1_prime,v1,1-parity])
        Vprime_flag = tuple([f0,e0,v0_prime,f1,e1,v1_prime,1-parity])
        visit_potential = [Fprime_flag,Eprime_flag,Vprime_flag]
        for i in visit_potential:
            if i in visit_set:
                pass
            else:
                queue.append(i)
                visit_set.add(i)
    return True,[sym_parity,sym_n,fix_items,double_sym_flag,flag_maps,unique_set]
    #'''
     

def symmetries_with_tags(tensor_0,tensor_1,fast = False,unique_calc=False):
    #generate all flags of each tensor
    #over flag_0, and flag_1.
    Sym_vecs=[]
    flaglist_0 = np.array(np.where(tensor_0==1)).transpose()
    flaglist_1 = np.array(np.where(tensor_1==1)).transpose()

    unique_flag_set=set()
    #find the union of all of the flag maps that are unique under each symmetry
    #in order to find the set of flags that are unique under *all* symmetries.
    face_set0 = np.sum(np.sum(tensor_0,axis=0),axis=0)/2
    face_set1 = np.sum(np.sum(tensor_1,axis=0),axis=0)/2

    flag_0=flaglist_0[0]
    face0 = face_set0[flag_0[2]]#v e f

    maps_0 = tensor_to_flag_matrix(tensor_0)
    maps_1 = tensor_to_flag_matrix(tensor_1)
    for i in range(len(flaglist_1)):
        face1 = face_set1[flaglist_1[i][2]]
        output,info = double_bfs(tensor_0,tensor_1,flag_0,flaglist_1[i],unique_calc)
        '''
        if unique_calc:
            output,info = double_bfs(tensor_0,tensor_1,flag_0,flaglist_1[i],unique_calc)
        else:
            output,info = symmetry_2(maps_0,maps_1,tensor_0,tensor_1,flag_0,flaglist_1[i],unique_calc)
        #'''
        #outputs are the same
        if face0==face1:
            if output:
                if fast:
                    return True
                #'''
                if unique_calc:
                    if len(unique_flag_set)==0:
                        unique_flag_set = set(info[-1])
                    else:
                        unique_flag_set = unique_flag_set.intersection(info[-1])
                #'''
                info.append((flag_0,flaglist_1[i]))
                Sym_vecs.append(info)

    if fast:
        return False
    if unique_calc:
        return Sym_vecs,unique_flag_set
    return Sym_vecs

def find_main_translation_maps(tensor,raw_sym_details):
    raw_sym_list, unique_flag_set = raw_sym_details#symmetries_with_tags(tensor,tensor)
    translation_maps=[]
    for i in raw_sym_list:
        mirror,order,data,double_sym_flag,flag_maps,unique_set,flags = i
        #translation maps + identity
        if (mirror and order==0) or order==3:
            if (flags[0]==double_sym_flag).all():
                translation_maps.append(flag_maps)
                #now i need to find the things I need to cut, and use this to find their maps
    #len(translation_maps) should always be 4.
    return translation_maps

def get_unique_set(graphs):
    #subset of wallpaper symmetry
    #goal is to find the faces, edges, and vertices that are unique under all maps.
    #I have as input the matrices
    unique_FEV_set = [set(),set(),set()]

    deriv_FEV_set = [set(),set(),set()]
    for i in range(3):
        for j in range(len(graphs[i])):
            if not(j in deriv_FEV_set[i]):#skip this one, we've already accounted for it.
                if not(j in unique_FEV_set[i]):# add it if we have not encountered it.
                    unique_FEV_set[i].add(j)
                #now we must add *all* points equivalent to it in deriv_FEV_set
                #AAA i can't think under this pressure
                #I need to traverse the graph.
                vertices_to_search = [j]
                vertices_visited = set()
                while True:
                    a = vertices_to_search.pop()
                    vertices_visited.add(a)
                    new_vertices = set(np.where(graphs[i][a]==1)[0])
                    diff = (new_vertices-vertices_visited)
                    if len(diff)==0:
                        #there are no new vertices to add
                        break
                    #otherwise, there are new vertices to add
                    vertices_to_search += list(diff)
                    #print(vertices_to_search,vertices_visited,new_vertices,diff)
                deriv_FEV_set[i]|=vertices_visited
                #if i==2:
    return unique_FEV_set

def wallpaper_symmetry(tensor,raw_sym_details):
    #this is explicity 2D, as the wallpaper group is 2D
    raw_sym_list,unique_flag_set = raw_sym_details#symmetries_with_tags(tensor,tensor,unique_calc=True)
    #unique_flag_set may be useful here?

    translation_maps = []
    fixed_items = []
    sym_type=[0,0,0,0,0]
    #this stores how many of each type of symmetry is involved in the tensor.
    #identity, translation, rotation, reflection, glide reflection
    #7 types total. Why?
    rotation_centers = []
    reflection_planes = []
    #stores the places of rotation centers, then compares them to existing mirror lines
    
    
    
    # the symmetries form a directed graph
    # I need to find not that.
    graphs = [np.zeros((tensor.shape[0],tensor.shape[0])),np.zeros((tensor.shape[1],tensor.shape[1])),np.zeros((tensor.shape[2],tensor.shape[2]))]
    #for i in range(20):
    #    print(i,np.sum(tensor[:,:,i])//2)
    for i in raw_sym_list:
        mirror,order,fix_items,double_sym_flag,flag_maps,unique_set,flags = i
        '''
        if flag_maps[2][15]==8:
            for k in i:
                print(k)
                print()
            print(1/0)
        #'''
        for j in range(3):
            L2 = list(flag_maps[j])
            for k in L2:
                graphs[j][k][flag_maps[j][k]] = 1
                graphs[j][flag_maps[j][k]][k] = 1
        if not(mirror):
            #reflection, glide reflection
            if order == 2:
                #reflection
                fixed_items.append(fix_items)
                reflection_planes.append(flag_maps)
                sym_type[3]+=1
            elif order == 0:
                #glide reflection
                #print(i)
                sym_type[4]+=1
            else:
                print("Symmetry type not found: 0")
                pass
        else:
            #identity, translation, rotation
            if order == 3:
                sym_type[0]+=1
                translation_maps.append(flag_maps)#honorary translation
                #identity
            elif order == 1:
                #rotation
                rotation_centers.append(flag_maps)
                sym_type[2]+=1
            elif order == 0:
                translation_maps.append(flag_maps)
                #translation
                #for j in i:
                #    print(j)#
                #print("\n"*3)
                sym_type[1]+=1
            else:
                #print(i)
                print("Symmetry type not found: 1")
    #anyway, i am here to find the connected groups
    unique_FEV_set = get_unique_set(graphs)# done!
    
    #print('\n'*5)
    #print(derivative_FEV_set)
    #print(unique_FEV_set[3])
    
    total_cells = sym_type[0]+sym_type[1]
    rot_total = sym_type[2]//total_cells
    mirror_total = (sym_type[3]+sym_type[4])//total_cells
    
    #print(sym_type)
    #print(mirror_planes)
    #for each rotation, do there exist items that are fixed under the rotation that are also fixed under the mirror?
    #rotation_types = [0,0]#on mirror, off mirror
    #we're labelling rotation centers
    fix_set = [set(),set(),set()]
    fix_set1 = [[],[],[]]
    #unique_flag_set
    #rotation_centers
    #print(rotation_centers)
    for i in rotation_centers:#unique_rotation_center_orders
        for j in range(3):
            v=i[j]
            v2 = list(v)
            for k in v2:
                if v[k]==k:
                    #this outputs the list of elements fixed under each rotation symmetry
                    fix_set[j].add(k)
                    fix_set1[j].append(k)
    
    #print(fix_set1)
    #print(fix_set)
    #print(unique_FEV_set)
    unique_rot_number = 0
    for i in range(3):
        unique_rot_number += len(fix_set[i]&unique_FEV_set[i])
    rotation_center_orders = []
    for i in range(len(fix_set)):
        for j in fix_set[i]:
            rotation_center_orders.append(fix_set1[i].count(j)+1)#+1 for the identity mapping
    rotation_center_orders.sort()
    rotation_center_orders = rotation_center_orders[::total_cells]#take one from each unit cell
    #print(rotation_center_orders)
    #print(reflection_planes)
    rotation_order = rot_total+1
    '''
    fix_set2 = [set(),set(),set()]
    for i in reflection_planes:
        for j in range(3):
            v=i[j]
            for k in fix_set[j]:
                if v[k]==k:
                    fix_set2[j].add(k)
    #'''
    #interacting reflection planes
    #for i in range(len(reflection_planes)):
    #    for j in range(i,len(reflection_planes)):
    #        #detect if there is something they both share
    #rotation_types[0] = sum(list(map(len,fix_set2)))//total_cells
    #rotation_types[1] = (sum(list(map(len,fix_set)))//total_cells) - rotation_types[0]
    if mirror_total>0:
        point_group = "D"+ str(rot_total+1)
        if rotation_order==1:
            #pm,cm,pg
            if sym_type[3]>0:#mirrors, not glides
                #pm,cm
                #pm,cm section
                translated_images = []
                for j in translation_maps:
                    item0,item1 = fixed_items[0][0]
                    f0,e0,v0 = j[0][item0[0]],j[1][item0[1]],j[2][item0[2]]
                    translated_images.append((f0,e0,v0))
                
                q = True
                for mirror_fixed in fixed_items:
                    p = False
                    for i in mirror_fixed:
                        if tuple(i[0]) in translated_images:
                            p = True
                            break
                    if not(p):
                        q = False
                #now see what mirrors it hits?
                #pm,cm section
                if q:
                    return "cm",point_group
                else:
                    return "pm",point_group
            else:
                return "pg",point_group
        elif rotation_order==2:
            #cmm,pmm,pmg,pgg
            if unique_rot_number==2:
                #pmg,pgg
                if sym_type[3]>0:#mirrors, not glides
                    return "pmg",point_group
                    #or cmm, apparently
                    #so I need something else to differentiate them.
                else:
                    return "pgg",point_group
            elif unique_rot_number==3:
                return "cmm",point_group
            else:
                return "pmm",point_group
        elif rotation_order==3:
            #p3m1,p31m
            if unique_rot_number==2:
                return "p31m",point_group
            else:
                return "p3m1",point_group
        elif rotation_order==4:
            #p4m,p4g
            if unique_rot_number==2:
                return "p4g",point_group
            else:
                return "p4m",point_group
        elif rotation_order==6:
            #p6m
            return "p6m",point_group
        else:
            print("Error, wallpaper not found")
    else:
        point_group = "C"+ str(rot_total+1)
        if rotation_order==1:
            return "p1",point_group
        elif rotation_order==2:
            return "p2",point_group
        elif rotation_order==3:
            return "p3",point_group
        elif rotation_order==4:
            return "p4",point_group
        elif rotation_order==6:
            return "p6",point_group
        else:
            print("Error, wallpaper not found")

def tensor_descendant(embedding_tensor,graph_details,raw_sym_details,face_to_split,edge_1,edge_2,vertex):
    #pos_init
    #modify pos_init to get the positions of the new vertices
    #use map to find the list of four faces to split, and 4 pairs of edges.
    translation_maps = find_main_translation_maps(embedding_tensor,raw_sym_details)
    #ordered vertex, edge, face
    #edge_pairs=[]
    #faces=[]
    #vertices=[]
    pos_init, X_coord_adj, Y_coord_adj = graph_details

    pos_2 = np.array(pos_init)
    X2 = np.array(X_coord_adj)
    Y2 = np.array(Y_coord_adj)
    graph_detail2 = (pos_2,X2,Y2)

    tensor_processing = np.array(embedding_tensor)
    for i in translation_maps:
        e0 = i[1][edge_1]
        e1 = i[1][edge_2]
        #edge_pairs.append([e0,e1])#edge
        f0 = i[2][face_to_split]
        v0 = i[0][vertex]
        #print(f0,e0,e1,v0)
        #faces.append(f0)#face
        #vertices.append(v0)
        tensor_processing,graph_detail2 = tensor_slice(tensor_processing,graph_detail2,f0,e0,e1,v0)
    #abcdef
    return tensor_processing,graph_detail2

def tensor_to_adj(tensor):
    #goal is to get the tensor to an adjancency matrix 
    V = len(tensor)
    Adj_matrix = np.zeros((V,V))
    for i in range(V):
        for j in range(i+1,V):
            indicator = np.sum(tensor[i]*tensor[j])
            if indicator:
                Adj_matrix[i][j]=1
                Adj_matrix[j][i]=1
    return Adj_matrix
    
def tensor_compress(tensor):
    A=[]
    for i in range(tensor.shape[1]):#iterating over edges
        a,b=np.where(tensor[:,i])
        A.append([list(set(a)),list(set(b))])
    return A

def tensor_decompress(Arr):
    #get tensor dimensions
    b = len(Arr)
    a,c = np.max(Arr,axis=(0,2))+1
    #use the max value in faces, vertices to retrieve the initial 
    Z = np.zeros((a,b,c))
    for i in range(len(Arr)):
        a2,c2 = Arr[i]
        for i2 in a2:
            for j2 in c2:
                Z[i2][i][j2]=1
    return Z

def relax_unit_vec(A,Z,X_coord_adj,Y_coord_adj):
    # doesn't work yet - need to scale each axis and shear...
    # not sure how, id need to *solve* for it, but the iterative approach doesn't work either.
    # maybe i make the atoms repel each other?
    lookup = np.array(np.where(A==1)).transpose()
    lookup_dict = dict()
    for i in lookup:
        if not(i[0] in lookup_dict):
            lookup_dict[i[0]]=[]
        lookup_dict[i[0]].append(i[1])
    M = np.zeros((2,2))
    for i in range(len(A)):
        v = lookup_dict[i]
        for j in v:
            delta = Z[j]-Z[i] + np.array([X_coord_adj[i][j],Y_coord_adj[i][j]])
            #l_delta = np.dot(delta,delta)
            #print(l_delta)
            M += np.tensordot(delta,delta,axes = 0)
    V = np.linalg.inv(scipy.linalg.sqrtm(M))
    #next, determine the unit cell scaling
    l_avg = 0
    l2_avg = 0
    L0 = 1.42481
    
    for i in range(len(A)):
        v = lookup_dict[i]
        for j in v:
            delta = Z[j]-Z[i] + np.array([X_coord_adj[i][j],Y_coord_adj[i][j]])
            delta2 = delta @ V
            L = np.sum((delta2)**2)**0.5
            l_avg += L
            l2_avg += L**2
    gamma = L0 * l_avg / l2_avg
    return V * gamma

def relax_tensor_2(tensor,graph_details):
    Z1, X_coord_adj,Y_coord_adj = graph_details
    A = np.array(tensor_to_adj(tensor))#adjacency matrix
    b_vec1 = np.sum(X_coord_adj + (1j * Y_coord_adj),axis=0)
    A1 = A - 3*np.identity(len(A))
    x = np.linalg.lstsq(A1,b_vec1,rcond=None)[0]
    Z = np.array([x.real,x.imag]).transpose()
    #find difference vectors
    skew_matrix = relax_unit_vec(A,Z,X_coord_adj,Y_coord_adj)
    return Z, skew_matrix

def display_tensor(tensor,graph_details): 
    Z, X_coord_adj,Y_coord_adj = graph_details
    
    def display_out(Z,skew_matrix,X_coord_adj,Y_coord_adj):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect('equal', adjustable='box')
        Z = Z @ skew_matrix.transpose()
        u = skew_matrix[:,0]#X
        v = skew_matrix[:,1]#Y
        plt.scatter(Z.transpose()[0],Z.transpose()[1])
        plt.scatter(Z.transpose()[0]+u[0],Z.transpose()[1]+u[1])
        plt.scatter(Z.transpose()[0]+v[0],Z.transpose()[1]+v[1])
        plt.scatter(Z.transpose()[0]+u[0]+v[0],Z.transpose()[1]+u[1]+v[1])
        
        for i in range(len(Z)):
            plt.text(Z[i][0],Z[i][1],str(i))
            plt.text(Z[i][0]+u[0],Z[i][1]+u[1],str(i))
            plt.text(Z[i][0]+v[0],Z[i][1]+v[1],str(i))
            plt.text(Z[i][0]+u[0]+v[0],Z[i][1]+u[1]+v[1],str(i))
        
        #displays edges
        #'''
        for i in range(len(M2)):
            for j in range(len(M2)):
                if M2[i][j]==1:#M2 is defined where?
                    #get its average position
                    #would be Z[j], but we have to correct it.
                    #averaging vector component, corrected for loops.
                    dj = np.array([X_coord_adj[i][j],Y_coord_adj[i][j]])
                    k2 = Z[j]+ (dj @ skew_matrix.transpose())
                    x0,y0=Z[i]
                    x1,y1=k2
                    plt.plot([x0,x1],[y0,y1],c='darkblue')#plot the edges
                    plt.plot([x0+u[0],x1+u[0]],[y0+u[1],y1+u[1]],c='darkblue')
                    plt.plot([x0+v[0],x1+v[0]],[y0+v[1],y1+v[1]],c='darkblue')
                    plt.plot([x0+u[0]+v[0],x1+u[0]+v[0]],[y0+u[1]+v[1],y1+u[1]+v[1]],c='darkblue')
        #'''
        plt.show()
    
    #tells if the connection from i->j crosses a boundary
    
    #alpha = 0.5
    #alpha = 0.9
    vertex_num = tensor.shape[0]
    M2 = np.array(tensor_to_adj(tensor))
    #forcing translations is not necesary here.
    #I don't have to set X_coord_adj,Y_coord_adj

    #'''
    #now the connections are set in stone
    #Z,skew_matrix = relax_tensor(tensor,graph_details)
    Z, skew_matrix = relax_tensor_2(tensor,graph_details)
    display_out(Z,skew_matrix,X_coord_adj,Y_coord_adj)

def save_tensor(tensor,graph_details,raw_sym_details,filename):
    t_comp = tensor_compress(tensor)
    Z, skew_matrix = relax_tensor_2(tensor,graph_details)
    pos, X_coord_adj, Y_coord_adj = graph_details
    #raw_sym_details?
    #print(raw_sym_details)
    tot_file = [np.array(t_comp).tolist(),np.array(Z).tolist(), np.array(X_coord_adj).tolist(), np.array(Y_coord_adj).tolist(),raw_sym_details]
    filetype = filename[-3:]
    if filetype == 'pkl':
        with open(filename, 'wb') as file:
            pickle.dump(tot_file,file)
    elif filetype == 'yml':
        with open(filename, 'w') as file:
            yaml.dump(tot_file, file, default_flow_style=False)

def load_tensor(filename):
    filetype = filename[-3:]
    if filetype == 'pkl':
        with open(filename, 'rb') as file:
            tot_file = pickle.load(file)
    elif filetype == 'yml':
        with open(filename, 'rb') as file:
            tot_file = yaml.safe_load(file)
    t_comp, pos, X_coord_adj, Y_coord_adj = tot_file
    tensor_out = tensor_decompress(t_comp)
    raw_sym_details = symmetries_with_tags(tensor_out,tensor_out,fast = False,unique_calc=True)
    graph_details = [np.array(pos), np.array(X_coord_adj), np.array(Y_coord_adj)]
    return tensor_out,graph_details,raw_sym_details

def generate_prof_file(tensor,graph_details,sym,filename):
    s=""
    output = np.array(np.where(tensor_to_adj(tensor))).transpose()
    Z,skew_matrix = relax_tensor_2(tensor,graph_details)
    #output *only* the primitive cell?
    #g = Z.transpose() #that part assumes an identity skew tensor, but i shouldn't be meddling with that.
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')
    plt.scatter(g[0],g[1])
    plt.show()
    #'''
    for i in output:
        s+="%d %d\n"%(i[0],i[1])
    s+="\n"
    for i in Z:
        s+="C %f %f 0.0\n"%(i[0],i[1])
    s+="\n"
    s+="{A00} {A10} 0.0 True\n".format(A00 = skew_matrix[0][0],A10= skew_matrix[1][0])
    s+="{A01} {A11} 0.0 True\n".format(A01 = skew_matrix[0][1],A11= skew_matrix[1][1])
    s+="0.0 0.0 1.0 False\n"
    s+=str(sym[0])
    f = open(filename,"w+")
    f.write(s)
    f.close()

def search_bfs(bfs_in,tensor_list):
    tensor_comp, graph_details_init, raw_sym_details = bfs_in
    tensor = tensor_decompress(tensor_comp)
    sym,unique_flag_set = raw_sym_details#symmetries_with_tags(tensor,tensor,fast = False,unique_calc=True)
    #unique_flag_set
    #can i remove the duplicate vertices?
    #'''
    flag_set_2=set()
    for i in unique_flag_set:
        v,e,f = i
        flag_set_2.add((e,f))
    unique_flag_set = flag_set_2
    #vertex = np.array(np.where(tensor[:,edge1,:])).transpose()[0,0]
    #'''
    #print(len(unique_flag_set))
    time_compare_last = 0
    time_compare_first = 0
    completed = []
    for flags in unique_flag_set:
        #print(flags)
        face = flags[1]
        edge1 = flags[0]
        #vertex = flags[0]
        edges_to_iterate = np.where(np.sum(tensor[:,:,face],axis=0))[0]
        #now we iterate over edges.
        for edge2 in edges_to_iterate:
            vertex = np.array(np.where(tensor[:,edge1,:])).transpose()[0,0]
            if edge1!=edge2 and not((face,edge2,edge1,vertex) in completed) and not((face,edge1,edge2,vertex) in completed):
                vertex = np.array(np.where(tensor[:,edge1,:])).transpose()[0,0]
                #print(i,edges_to_iterate[i2],edges_to_iterate[j2])
                #print(vertex)
                #print(flags)
                tensor_out,graph_details = tensor_descendant(tensor,graph_details_init,raw_sym_details,face,edge1,edge2,vertex)
                #print((face,edge1,edge2,vertex))
                completed.append((face,edge1,edge2,vertex))
                #print(i,edges_to_iterate[i2],edges_to_iterate[j2])
                #now we test to see if the tensor is in our list already
                #t0 = time.time()
                p = True
                for k in tensor_list:
                    tensor1 = tensor_decompress(k[0])
                    q_bool = quick_compare(tensor_out,tensor1)
                    if q_bool: 
                        #pool.submit(worker)
                        #comparing by eigenvalues is *slower* here?
                        if symmetries_with_tags(tensor_out,tensor1,fast = True):
                            p = False
                            #print(len(sym_vecs))
                            #it is congruent with at least one of the other tensors in the bunch.
                            break
                #t1 = time.time()
                if p:
                    #if the tensor is not in the list, add it.
                    raw_sym_details = symmetries_with_tags(tensor_out,tensor_out,fast = False,unique_calc=True)
                    tensor_list.append((tensor_compress(tensor_out),graph_details,raw_sym_details))
                #t2 = time.time()
                #time_compare_first += t1-t0
                #time_compare_last += t2-t1
            #'''
    #we modified the tensor list. this is our "output" per se.
    #print(time_compare_last,time_compare_first)
    return tensor_list

def quick_compare(tensor1,tensor2):
    face_arr1 = list(np.sum(np.sum(tensor1,axis=0),axis=0)/2)
    face_arr2 = list(np.sum(np.sum(tensor2,axis=0),axis=0)/2)
    face_arr1 = list(map(int,face_arr1))
    face_arr2 = list(map(int,face_arr2))
    face_arr1.sort()
    face_arr2.sort()
    if face_arr1==face_arr2:
        return True
    else:
        return False
    #if returns False, tensors are not the same
    #if returns true, they definetly are not.

def test_duplicate(face_arr1,raw_sym_details):
    face_arr1=face_arr1[::4]
    fdict = dict()
    for i in range(len(face_arr1)):
        if not(face_arr1[i] in fdict):
            fdict[face_arr1[i]]=0
        fdict[face_arr1[i]]+=1
    A=[]
    for i in fdict:
        A.append(fdict[i])
    x = A[0]
    for i in A[1:]:
        x = np.gcd(x,i)
    if x==1:
        return False#is not duplicate.
    
    raw_sym_list, unique_flag_set = raw_sym_details#symmetries_with_tags(tensor,tensor)
    count = 0
    for i in raw_sym_list:
        mirror,order,data,double_sym_flag,flag_maps,unique_set,flags = i
        #translation maps + identity
        if (mirror and order==0) or order==3:
            count+=1
    if count==4:
        return False
    else:
        #count is greater than 4, there are more than 4 cells.
        return True

def search_all(depth):
    #given a list of tensors, output the list of the next set.
    tensor_list = []
    if depth == 1:
        graph_details = generate_graph_details()
        tensor_init = generate_init_tensor(1)
        raw_sym_details = symmetries_with_tags(tensor_init,tensor_init,fast = False,unique_calc=True)
        tensor_list.append((tensor_compress(tensor_init),graph_details,raw_sym_details))
    else:
        tensor_list_prev = search_all(depth-1)
        for i in tensor_list_prev:
            tensor_list = search_bfs(i,tensor_list)
    #'''
    name_dict=dict()
    for j in tensor_list:
        tensor = tensor_decompress(j[0])
        graph_details = j[1]
        raw_sym_details = j[2]
        sym = wallpaper_symmetry(tensor,raw_sym_details)
        face_arr1 = list(np.sum(np.sum(tensor,axis=0),axis=0)/2)
        face_arr1 = list(map(int,face_arr1))
        face_arr1.sort()
        #test to see if it's a duplicate
        if not(test_duplicate(face_arr1,raw_sym_details)):#sym[0]=="pm,cm":
            name = str(face_arr1[::4])[1:-1].replace(',','-')
            if name in name_dict:
                filename = name + '_'+str(name_dict[name])
                name_dict[name]+=1
            else:
                filename = name
                name_dict[name]=1
            filename=filename.replace(' ','')
            print(filename)
            #save_tensor(tensor,graph_details,raw_sym_details,sym[0]+'//'+filename +'.pkl')#sym[0]+'//'+
            generate_prof_file(tensor,graph_details,sym,sym[0]+'//'+filename+".txt")#not a cif file
            tensor_only_output(tensor,raw_sym_details,sym[0]+'//'+filename+".fev")
    #'''
    #print(round(time.time()-t0,5),depth,len(tensor_list))
    return tensor_list

def tensor_only_output(tensor,raw_sym_details,filename):#Edu's format
    tensor = primitive_tensor(tensor,raw_sym_details)
    tensor = tensor.transpose()
    l,w,h = tensor.shape
    s = "\t%s\t%s\t%s"%(l,w,h)
    s += "\n\t"
    #tensor_decompress(t_comp)
    tensor_string = '\n\t'.join(list(map(str,tensor.astype(int).flatten())))
    #print(tensor_string)
    '''
    flags = np.array(np.where(tensor==1)).transpose()
    for i in flags:
        s += "%s, %s, %s\n"%tuple(i)
    #'''
    f = open(filename,"w+")
    f.write(s+tensor_string)
    f.close()

def primitive_tensor(tensor,raw_sym_details):
    raw_sym_list, unique_flag_set = raw_sym_details#symmetries_with_tags(tensor,tensor)
    translation_maps=[]
    graphs = [np.zeros((tensor.shape[0],tensor.shape[0])),np.zeros((tensor.shape[1],tensor.shape[1])),np.zeros((tensor.shape[2],tensor.shape[2]))]
    for i in raw_sym_list:
        mirror,order,data,double_sym_flag,flag_maps,unique_set,flags = i
        #translation maps + identity
        if (mirror and order==0) or order==3:
            for j in range(3):
                L2 = list(flag_maps[j])
                for k in L2:
                    graphs[j][k][flag_maps[j][k]] = 1
                    graphs[j][flag_maps[j][k]][k] = 1
            translation_maps.append(flag_maps)
    #make a graph
    down_convert_map = [dict(),dict(),dict()]
    unique_FEV_set = get_unique_set(graphs)
    for i in range(3):
        k=0
        for j in unique_FEV_set[i]:
            down_convert_map[i][j]=k
            k+=1
    Z = np.zeros((len(unique_FEV_set[0]),len(unique_FEV_set[1]),len(unique_FEV_set[2])))
    flaglist = np.array(np.where(tensor==1)).transpose()
    for i in flaglist:
        a,b,c=i
        s0 = list(set(np.where(graphs[0][a]==1)[0]) & unique_FEV_set[0])[0]
        s1 = list(set(np.where(graphs[1][b]==1)[0]) & unique_FEV_set[1])[0]
        s2 = list(set(np.where(graphs[2][c]==1)[0]) & unique_FEV_set[2])[0]

        k0 = down_convert_map[0][s0]
        k1 = down_convert_map[1][s1]
        k2 = down_convert_map[2][s2]
        Z[k0][k1][k2]=1
    return Z
    #return translation_maps
#'''

#from cProfile import Profile
#from pstats import SortKey, Stats

t0=time.time()
output = search_all(1)
'''
with Profile() as profile:
    output = search_all(2)
    (
    Stats(profile)
    .strip_dirs()
    .sort_stats(SortKey.CUMULATIVE)
    .print_stats()
    )
#'''
t1=time.time()
print(t1-t0)
#output = wallpaper_symmetry(tensor_init)
#print(len(output))
#'''

'''
tensor_test = load_tensor('4,4,4,9,9.pkl')
display_tensor(tensor_test[0],tensor_test[1])
#primitive = primitive_tensor(tensor_test[0])
#save_tensor(tensor_test[0],tensor_test[1],'4,8.yml')#sym[0]+'//'+
#tensor_test = load_tensor('4,8.yml')
#display_tensor(tensor_test[0],tensor_test[1])

#tensor_only_output(primitive,'3411_cm_primitive.fev')
#out = wallpaper_symmetry(tensor_test[0])
#generate_prof_file(tensor_test[0],tensor_test[1],out[0],'4,4,6,6,10.txt')
#'''
