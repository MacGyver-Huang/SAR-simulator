//
//  vertexlist.h
//  PASSEDv4
//
//  Created by Steve Chiang on 11/15/22.
//  Copyright (c) 2022 Steve Chiang. All rights reserved.
//

#ifndef vertexlist_h
#define vertexlist_h

//+---------------------------+
//|        Vertex List        |
//+---------------------------+
template<typename T>
class VertexList {
public:
	/**
	 * Deafult constructor
	 */
	VertexList(){
		_L.resize(0);
//		_L.reserve(1000000);
	}
	/**
	 * Operator overloading [] (editable)
	 * @param[in]   i    Index
	 * @return Return the list on certain index, i
	 */
	VEC<T>& operator[](const size_t i){ return _L[i]; }
	/**
	 * Operator overloading []
	 * @param[in]   i    Index
	 * @return Return the list on certain index, i
	 */
	const VEC<T>& operator[](const size_t i)const{ return _L[i]; }
	/**
	 * Push back to this vertex list
	 * @param[in]   in  (VEC<T>) The position vector
	 */
	void push_back(const VEC<T>& in){ _L.push_back(in); }
	/**
	 * Get the size of this list
	 */
	size_t size(){ return _L.size(); }
	/**
	 * Display all memeber variables on the console
	 */
	void Print(){
		cout<<"+-----------------+"<<endl;
		cout<<"|   Vertex List   |"<<endl;
		cout<<"+-----------------+"<<endl;
		for(size_t i=0;i<_L.size();++i){
			cout<<"#"<<i<<" : "; _L[i].Print();
		}
	}
private:
	vector<VEC<T> > _L;			// Vertex index
};

#endif
