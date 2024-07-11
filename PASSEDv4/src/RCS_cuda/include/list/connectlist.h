//
//  connectlist.h
//  PASSEDv4
//
//  Created by Steve Chiang on 11/15/22.
//  Copyright (c) 2022 Steve Chiang. All rights reserved.
//

#ifndef connectlist_h
#define connectlist_h

#include <list/linkedlist.h>


//+---------------------------+
//|       Connect List        |
//+---------------------------+
class ConList {
public:
	/**
	 * Default consturctor
	 */
	ConList(){
		_L.resize(0);
//		_L.reserve(1000000);
	}
	/**
	 * Operator overloading [] (editable)
	 * @param[in]   i    Index
	 * @return Return the list on certain index, i
	 */
	LinkedList<size_t>& operator[](const size_t i){ return _L[i]; }
	/**
	 * Operator overloading []
	 * @param[in]   i    Index
	 * @return Return the list on certain index, i
	 */
	const LinkedList<size_t>& operator[](const size_t i)const{ return _L[i]; }
	/**
	 * Push back to this connect list
	 * @param[in]   idx      The list index we want to insert
	 * @param[in]   idx_poly The polygon index
	 */
	void push_back(const size_t idx, const size_t idx_poly){
		// push empty list
		if(_L.size() == 0 || _L.size()-1 < idx){
			for(size_t i=_L.size();i<=idx;++i){
				_L.push_back( LinkedList<size_t>() );
			}
		}
		// add element
		_L[idx].push_back(idx_poly);
	}
	/**
	 * Get the size of this connect list
	 */
	size_t size()const{ return _L.size(); }
	size_t size(){ return _L.size(); }
	/**
	 * Display all memeber variables on the console
	 */
	void Print()const{
		cout<<"+-----------------+"<<endl;
		cout<<"|  Connect List   |"<<endl;
		cout<<"+-----------------+"<<endl;
		for(size_t j=0;j<_L.size();++j){	// same size with Vertex List
			cout<<"#"<<j<<" : "; _L[j].Print();
		}
	}
	void Print(){
		cout<<"+-----------------+"<<endl;
		cout<<"|  Connect List   |"<<endl;
		cout<<"+-----------------+"<<endl;
		for(size_t j=0;j<_L.size();++j){	// same size with Vertex List
			cout<<"#"<<j<<" : "; _L[j].Print();
		}
	}
private:
	// Ref: https://en.wikipedia.org/wiki/Polygon_mesh
	vector<LinkedList<size_t> > _L;		// Connect list (called Mesh-Vertex meshes)
};

#endif
