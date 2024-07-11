//
//  polygonlist.h
//  PASSEDv4
//
//  Created by Steve Chiang on 11/15/22.
//  Copyright (c) 2022 Steve Chiang. All rights reserved.
//

#ifndef polygonlist_h
#define polygonlist_h

#include <vector>

using namespace std;

//+---------------------------+
//|       Polygon index       |
//+---------------------------+
class Polygon {
public:
	/**
	 * Default constructor
	 */
	Polygon(){};
	/**
	 * Constructor with values
	 * @param[in]   IV0   1st vertice index
	 * @param[in]   IV1   2nd vertice index
	 * @param[in]   IV2   3rd vertice index
	 */
	Polygon(const size_t IV0, const size_t IV1, const size_t IV2){ _iV0=IV0; _iV1=IV1; _iV2=IV2; }
	/**
	 * Get 1st vertice index
	 */
	size_t& IV0(){ return _iV0; }
	/**
	 * Get 2nd vertice index
	 */
	size_t& IV1(){ return _iV1; }
	/**
	 * Get 3rd vertice index
	 */
	size_t& IV2(){ return _iV2; }
	/**
	 * Get 3 vertex index
	 * @return Return a vector store 3 vertex index
	 */
	vector<size_t> IV(){
		vector<size_t> iv = {_iV0, _iV1, _iV2};
		return iv;
	}
	/**
	 * Display all memeber variables on the console
	 */
	void Print(){
		cout<<"["<<_iV0<<", "<<_iV1 <<", "<<_iV2<<"]"<<endl;
	}
	/**
	 * Output stream
	 */
	friend ostream &operator<<(ostream &os, const Polygon& in) {
		os<<"["<<in._iV0<<", "<<in._iV1 <<", "<<in._iV2<<"]"<<endl;
		return os;
	}
private:
	size_t _iV0, _iV1, _iV2;	// Vertex index with respective to the ListVertex
};


//+---------------------------+
//|       Polygon List        |
//+---------------------------+
class PolygonList {
public:
	/**
	 * Default constructor
	 */
	PolygonList(){
		_L.resize(0);
//		_L.reserve(1000000);
	}
	/**
	 * Operator overloading [] (editable)
	 * @param[in]   i   Index
	 * @return Return the Poly class in index, i
	 */
	Polygon& operator[](const size_t i){ return _L[i]; }
	/**
	 * Operator overloading []
	 * @param[in]   i   Index
	 * @return Return the Poly class in index, i
	 */
	const Polygon& operator[](const size_t i)const{ return _L[i]; }
	/**
	 * Push back the vertex by giving 3 vertex index
	 * @param[in]   idx_v0   1st vertice index
	 * @param[in]   idx_v1   2nd vertice index
	 * @param[in]   idx_v2   3rd vertice index
	 */
	void push_back(const size_t& idx_v0, const size_t& idx_v1, const size_t& idx_v2){
		Polygon py(idx_v0, idx_v1, idx_v2);
		_L.push_back( py );
	}
	/**
	 * Get the size of this polygon list
	 * @return Return the list size
	 */
	size_t size(){ return _L.size(); }
	/**
	 * Display all memeber variables on the console
	 */
	void Print(){
		cout<<"+-----------------+"<<endl;
		cout<<"|  Polygon List   |"<<endl;
		cout<<"+-----------------+"<<endl;
		for(size_t j=0;j<_L.size();++j){	// same size with Vertex List
			cout<<"#"<<j<<" : ["<<_L[j].IV0()<<", "<<_L[j].IV1()<<", "<<_L[j].IV2()<<"]"<<endl;
		}
	}
private:
	vector<Polygon> _L;	// Polygon list (store all polygon index, Poly)
};



#endif
