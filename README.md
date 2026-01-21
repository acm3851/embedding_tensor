# Embedding Tensor (Beta)
This code contains a number of functions that are useful in determining equivalence of tensors. This code is in beta.

### generate_init_tensor(num_polygons)
generates the tensor corresponding to 6^num_polygons.
6^1 is used as the base of the search algorithm
usage:
```
generate_init_tensor(1) # generates the tensor for h6^1
generate_init_tensor(2) # generates the tensor for h6^4
```
### tensor_slice(embedding_tensor,graph_detail,face_to_split,edge_1,edge_2,vertex)
cuts a face through edges 1 and 2 on the given tensor, using a vertex to determine orientation.

### generate_graph_details()
generates the base positions, X and Y coordinate matrices for 6^1.

### double_flag_to_str(item0,item1)
an auxillary function for double_bfs that combines two flag representations in a way that is hashable.

### double_bfs(tensor_0,tensor_1,flag_0,flag_1,unique_calc=False)
the main function that determines symmetry, as implemented in the paper.
arguments: two tensors tensor_0 and tensor_1, the flags that they start on (flag_0,flag_1), and unique_calc, which takes extra time to collect a unique set of flags.

### tensor_to_flag_matrix(tensor)

### symmetry_2(maps_0,maps_1,tensor_0,tensor_1,flag_0,flag_1,unique_calc=False)
A secondary function that determines the symmetry, the same as double_bfs. Cleaner, but slightly slower.
arguments: two tensors tensor_0 and tensor_1, the flags that they start on (flag_0,flag_1), and unique_calc, which takes extra time to collect a unique set of flags.

### symmetries_with_tags(tensor_0,tensor_1,fast = False,unique_calc=False)
the wrapper function for double_bfs, that iterates through all th flags of both tensors.
the "fast" boolean determines if the goal is to collect all symmetries, or to determine if the tensors are equivalent.

### find_main_translation_maps(tensor,raw_sym_details)
finds the translation maps corresponding to the 2x2 supercell that is used during calculation

### get_unique_set(graphs)
this function finds the the faces, edges, and vertices that are unique under all maps.

### wallpaper_symmetry(tensor,raw_sym_details)
finds the wallpaper group of the given tensor and the "raw_sym_details" structure generated from running the symmetry algorithm on it.

### tensor_descendant(embedding_tensor,graph_details,raw_sym_details,face_to_split,edge_1,edge_2,vertex)
calls tensor_slice, but equally on all four 2x2 unit cells.

### tensor_to_adj(tensor)
takes in the tensor, outputs the corresponding adjacency matrix.

### tensor_compress(tensor)
returns a compressed version of the tensor using the edges to enumerate it. this compressed version is of size O(4E)

### tensor_decompress(Arr)
returns a decompressed version of the tensor from the edge-based encoding.

### relax_unit_vec(A,Z,X_coord_adj,Y_coord_adj)
part of the algorithm that outputs an image of the tensor generated. it takes the adjacency matrix, fractional coordinates, and the connections above and below the unit cell, and outputs approximate unit cell vectors.

### relax_tensor_2(tensor,graph_details)
part of the algorithm that outputs an image of the tensor generated. It outputs the fractional coordinates and relaxed unit cell vectors.

### display_tensor(tensor,graph_details)
graphing module, displays a pyplot image of the corresponding structure.
```
tensor_test = load_tensor('4,4,4,9,9.pkl')
display_tensor(tensor_test[0],tensor_test[1])
```

### save_tensor(tensor,graph_details,raw_sym_details,filename)
saving module, saves all the relavant details, including extra symmetry information. The file can be rather large.

### load_tensor(filename)
inverse of above, loads the tensor with all symmetry information preserved.

### generate_prof_file(tensor,graph_details,sym,filename)
outputs in a special format that includes the adjacency matrix, approximate atom locations, and unit cell vectors.
format consists of information sections separated by a newline
1. lists of vertices connected by an edge
0 5
0 24
0 39
1 4
1 10
1 11
...
2. Atomic locations. Atom type, then X, Y, Z.

C -0.320517 -0.460517 0.0
C -0.246379 -0.086379 0.0
C 0.179483 -0.460517 0.0
C 0.253621 -0.086379 0.0
...
3. Unit cell vectors along with connectivity. (As in, do they connect to the opposite side?)
13.71663521798131 7.731664348560368 0.0 True
7.731664348560369 13.716635217981315 0.0 True
0.0 0.0 1.0 False
4. Wallpaper Symmetry
p1

```
tensor_test = load_tensor('4,4,4,9,9.pkl')
out = wallpaper_symmetry(tensor_test[0])
generate_prof_file(tensor_test[0],tensor_test[1],out[0],'4,4,6,6,10.txt')
```
### search_bfs(bfs_in,tensor_list)
main component of the search algorithm for generating new structures. It takes in a list of tensors with N faces, and outputs the next set of unique tensors for N+1 faces.

### quick_compare(tensor1,tensor2)
method to speed up searches by only comparing against tensors that are known to have the same types of faces.

### test_duplicate(face_arr1,raw_sym_details)
tests to see if a tensor is a primitive cell or not.

### tensor_only_output(tensor,raw_sym_details,filename)
outputs the tensor in a stream of zeros and ones.

### primitive_tensor(tensor,raw_sym_details)
returns a primitve tensor from a given tensor which may have multiple translation symmetries.
the resuling tensor is not guaranteed to follow the sum rule and may have unintended behavior.

### search_all(depth)
wrapper for search_bfs, it recursively searches through tensors until it reaches all with the given depth.

Examples:
```
output = search_all(6) #writes to file all of the tensors from size 1 to 6.
```




