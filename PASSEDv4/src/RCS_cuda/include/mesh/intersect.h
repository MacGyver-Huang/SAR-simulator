//
//  intersect.h
//  PASSED2
//
//  Created by Steve Chiang on 3/26/18.
//  Copyright (c) 2018 Steve Chiang. All rights reserved.
//

#ifndef intersect_h
#define intersect_h

//
//  main.cpp
//  test_rectangle_segment_intersection_2D
//
//  Created by Steve Chiang on 3/26/18.
//  Copyright (c) 2018 Steve Chiang. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <basic/vec.h>
#include <bvh/triangle.h>

using namespace vec;


//#define SMALL_NUM  0.00000001 // anything that avoids division overflow
#define SMALL_NUM  0.00000000000001 // anything that avoids division overflow




using namespace std;


namespace intersect {
	// ==============================================
	// 2D Vector class
	// ==============================================
	template<typename T>
	class VEC2D {
	public:
		// Constructure
		VEC2D():_x(0),_y(0){};
		VEC2D(T a,T b):_x(a),_y(b){};
		// IO
		const T& x() const{return _x;};
		const T& y() const{return _y;};
		T& x(){return _x;};
		T& y(){return _y;};
		// Operator overloading
		template<typename T2> friend const VEC2D<T2> operator+(const VEC2D<T2>& L,const VEC2D<T2>& R){ return VEC2D<T2>( L._x+R._x, L._y+R._y ); }
		template<typename T2> friend const VEC2D<T2> operator-(const VEC2D<T2>& L,const VEC2D<T2>& R){ return VEC2D<T2>( L._x-R._x, L._y+R._y ); }
		template<typename T2> friend const VEC2D<T2> operator*(const T2& S,const VEC2D<T2>& R){ return VEC2D<T2>( S*R._x, S*R._y ); }
		// Misc.
		T length(){ return sqrt(_x*_x + _y*_y); }
		void Print(){
			cout<<"Vector(2D) : ("<<std::setprecision(10)<<_x<<","<<_y<<")"<<endl;
		};
	private:
		double _x, _y;
	};
	
	// ==============================================
	// 2D Point class
	// ==============================================
	template<typename T>
	class POINT2D {
	public:
		// Constructor
		POINT2D():_x(0),_y(0){};
		POINT2D(T a,T b):_x(a),_y(b){};
		// IO
		const T& x() const{return _x;};
		const T& y() const{return _y;};
		T& x(){return _x;};
		T& y(){return _y;};
		// Operator overloading
		template<typename T2> friend const POINT2D<T2> operator+(const POINT2D<T2>& L,const POINT2D<T2>& R){ return POINT2D<T2>( L._x+R._x, L._y+R._y ); }
		template<typename T2> friend const POINT2D<T2> operator+(const POINT2D<T2>& L,const VEC2D<T2>& R)  { return POINT2D<T2>( L._x+R.x(), L._y+R.y() ); }
		template<typename T2> friend const POINT2D<T2> operator-(const POINT2D<T2>& L,const VEC2D<T2>& R)  { return POINT2D<T2>( L._x-R._x, L._y-R._y ); }
		template<typename T2> friend const VEC2D<T2>   operator-(const POINT2D<T2>& L,const POINT2D<T2>& R){ return VEC2D<T2>  ( L._x-R._x, L._y-R._y ); }
		template<typename T2> friend const POINT2D<T2> operator*(const T2& S,const POINT2D<T2>& R){ return POINT2D<T2>( S*R._x, S*R._y ); }
		template<typename T2> friend bool operator==(const POINT2D<T2>& L, const POINT2D<T2>&  R){
			if((abs(L._x - R._x) + abs(L._y - R._y)) < SMALL_NUM){
				return true;
			}else{
				return false;
			}
		}
		// Misc.
		void Print(){
			cout<<"Point : ("<<std::setprecision(10)<<_x<<","<<_y<<")"<<endl;
		};
	private:
		double _x, _y;
	};
	
	// ==============================================
	// 2D Segment class
	// ==============================================
	template<typename T>
	class SEGMENT2D
	{
	public:
		// Constructure
		SEGMENT2D(){};
		SEGMENT2D(const POINT2D<T>& Start, const POINT2D<T>& End){ _S=Start; _E=End; }
		// IO
		const POINT2D<T>& S() const{return _S;};
		const POINT2D<T>& E() const{return _E;};
		POINT2D<T>& S(){return _S;};
		POINT2D<T>& E(){return _E;};
		// Misc.
		void Print(){
			cout<<"+-----------------+"<<endl;
			cout<<"|   Segment (2D)  |"<<endl;
			cout<<"+-----------------+"<<endl;
			cout<<"Start : "; _S.Print();
			cout<<"End   : "; _E.Print();
		}
	private:
		POINT2D<T> _S;	// Start point
		POINT2D<T> _E;	// End point
	};
	
	// ==============================================
	// 3D Segment class
	// ==============================================
	template<typename T>
	class SEGMENT
	{
	public:
		// Constructure
		SEGMENT(){};
		SEGMENT(const VEC<T>& Start, const VEC<T>& End){ _S=Start; _E=End; }
		template<typename T2>
		SEGMENT(const VEC<T2>& Start, const VEC<T2>& End){
			_S=VEC<T>(T(Start.x()), T(Start.y()), T(Start.z()));
			_E=VEC<T>(T(End.x()),   T(End.y()),   T(End.z()));
		}
		// IO
		const VEC<T>& S() const{return _S;};
		const VEC<T>& E() const{return _E;};
		VEC<T>& S(){return _S;};
		VEC<T>& E(){return _E;};
		// Misc.
		T Length(){ return (_E-_S).abs(); }
		void Print(){
			cout<<"+-----------------+"<<endl;
			cout<<"|   Segment (3D)  |"<<endl;
			cout<<"+-----------------+"<<endl;
			cout<<"Start : "; _S.Print();
			cout<<"End   : "; _E.Print();
		}
	private:
		VEC<T> _S;	// Start point
		VEC<T> _E;	// End point
	};
	
	// ==============================================
	// 3D Edge class
	// ==============================================
	template<typename T>
	class EdgeList
	{
	public:
		// Constructure
		EdgeList(){};
		EdgeList(const SEGMENT<T>& edge0, const SEGMENT<T>& edge1, const SEGMENT<T>& edge2){
			_edge[0] = edge0;
			_edge[1] = edge1;
			_edge[2] = edge2;
		}
		EdgeList(const TRI<T>& tri){
			_edge[0] = SEGMENT<T>(tri.V0(), tri.V1());
			_edge[1] = SEGMENT<T>(tri.V1(), tri.V2());
			_edge[2] = SEGMENT<T>(tri.V2(), tri.V0());
		}
		template<typename T2>
		EdgeList(const TRI<T2>& tri){
			_edge[0] = SEGMENT<T>(tri.V0(), tri.V1());
			_edge[1] = SEGMENT<T>(tri.V1(), tri.V2());
			_edge[2] = SEGMENT<T>(tri.V2(), tri.V0());
		}
		// IO
		const SEGMENT<T>& Edge(const size_t i) const{ return _edge[i]; };
		const SEGMENT<T>& Edge0() const{ return _edge[0]; };
		const SEGMENT<T>& Edge1() const{ return _edge[1]; };
		const SEGMENT<T>& Edge2() const{ return _edge[2]; };
		SEGMENT<T>& Edge(const size_t i){ return _edge[i]; };
		SEGMENT<T>& Edge0(){ return _edge[0]; };
		SEGMENT<T>& Edge1(){ return _edge[1]; };
		SEGMENT<T>& Edge2(){ return _edge[2]; };
		// Misc.
		T Length(const size_t i){ return _edge[i].Length(); }
		void Print(){
			cout<<"+--------------------+"<<endl;
			cout<<"|    EdgeList (3D)   |"<<endl;
			cout<<"+--------------------+"<<endl;
			cout<<"Edge 0 : "<<endl; _edge[0].Print();
			cout<<"Edge 1 : "<<endl; _edge[1].Print();
			cout<<"Edge 2 : "<<endl; _edge[2].Print();
		}
	private:
		SEGMENT<T> _edge[3]; // Edge set (ONLY 3 for triangle polygon)
	};
	
	// ==============================================
	// 2D Recangle class
	// ==============================================
	template<typename T>
	class RECT2D
	{
	public:
		// Constructure
		RECT2D(){};
		RECT2D(const POINT2D<T>& V0, const POINT2D<T>& V1, const POINT2D<T>& V2, const POINT2D<T>& V3){
			// *NOTE* It must be clockwise or counter clockwise continuly points
			_v[0] = V0;
			_v[1] = V1;
			_v[2] = V2;
			_v[3] = V3;
		}
		RECT2D(const POINT2D<T> V[4]){
			// *NOTE* It must be clockwise or counter clockwise continuly points
			for(size_t i=0;i<4;++i){
				_v[i] = V[i];
			}
		}
		// IO
		const POINT2D<T>& V(const size_t i) const{ return _v[i%4];};
		POINT2D<T>& V(const size_t i){ return _v[i%4]; }
		// Misc.
		void Print(){
			cout<<"+-------------------+"<<endl;
			cout<<"|   Rectangle (2D)  |"<<endl;
			cout<<"+-------------------+"<<endl;
			for(size_t i=0;i<4;++i){
				_v[i].Print();
			}
		}
	private:
		POINT2D<T> _v[4];
	};
	
	
	
	// ==============================================
	// Functions implement
	// ==============================================
	template<typename T>
	T dot2D(const VEC2D<T>& u, const VEC2D<T>& v){
		return u.x() * v.x() + u.y() * v.y();   // 2D dot product
	}
	
	template<typename T>
	T perp2D(const VEC2D<T>& u, const VEC2D<T>& v){
		return u.x() * v.y() - u.y() * v.x();   // 2D perp product
	}

	
	// cn_PnPoly(): crossing number test for a point in a polygon
	//      Input:   P = a point,
	//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
	//      Return:  0 = outside, 1 = inside
	// This code is patterned after [Franklin, 2000]
	template<typename T>
	int cn_PnPoly(const POINT2D<T>& P, const RECT2D<T>& Rect) {
		int cn = 0;    // the  crossing number counter
		
		// loop through all edges of the polygon
		for(size_t i=0; i<4; i++) {    // edge from V[i]  to V[i+1]
			if (((Rect.V(i).y() <= P.y()) && (Rect.V(i+1).y() > P.y()))     // an upward crossing
				|| ((Rect.V(i).y() > P.y()) && (Rect.V(i+1).y() <=  P.y()))) { // a downward crossing
															 // compute  the actual edge-ray intersect x-coordinate
				float vt = (float)(P.y()  - Rect.V(i).y()) / (Rect.V(i+1).y() - Rect.V(i).y());
				if (P.x() <  Rect.V(i).x() + vt * (Rect.V(i+1).x() - Rect.V(i).x())) // P.x < intersect
					++cn;   // a valid crossing of y=P.y right of P.x
			}
		}
		return (cn&1);    // 0 if even (out), and 1 if  odd (in)
		
	}
	
	// RectangleSegmentIntersection(): intersect a 2D segment with a convex polygon
	//    Input:  S = 2D segment to intersect with the convex polygon V[]
	//            n = number of 2D points in the polygon
	//            V[] = array of n+1 vertex points with V[n] = V[0]
	//      Note: The polygon MUST be convex and
	//                have vertices oriented counterclockwise (ccw).
	//            This code does not check for and verify these conditions.
	//    Output: *IS = the intersection segment (when it exists)
	//    Return: FALSE = no intersection
	//            TRUE  = a valid intersection segment exists
	template<typename T>
	T RectangleSegmentIntersection(const SEGMENT2D<T>& S, const RECT2D<T>& Rect, SEGMENT2D<T>& IS){
		if (S.S() == S.E()) {         // the segment S is a single point
									// test for inclusion of S.P0 in the polygon
			IS = S;						// same point if inside polygon
			return cn_PnPoly(S.S(), Rect);
		}
		
		T  tE = 0;						// the maximum entering segment parameter
		T  tL = 1;						// the minimum leaving segment parameter
		T  t, N, D;						// intersect parameter t = N / D
		VEC2D<T> dS = S.E()- S.S();		// the  segment direction vector
		VEC2D<T> e;						// edge vector

		
		for(size_t i=0; i<4; ++i) {		// process polygon edge V[i]V[i+1]
			
			e = Rect.V(i+1) - Rect.V(i);// It will auto detect the final point and warp to first
			
			N = perp2D(e, S.S()-Rect.V(i)); // = -dot(ne, S.P0 - V[i])
			D = -perp2D(e, dS);			// = dot(ne, dS)
			if (fabs(D) < SMALL_NUM) {	// S is nearly parallel to this edge
				if (N < 0) {			// P0 is outside this edge, so
					return -9999;		// S is outside the polygon
				} else {				// S cannot cross this edge, so
					continue;			// ignore this edge
				}
			}
			
			t = N / D;
			if (D < 0) {            // segment S is entering across this edge
				if (t > tE) {       // new max tE
					tE = t;
					if (tE > tL) {  // S enters after leaving polygon
						return -9999;
					}
				}
			} else {                // segment S is leaving across this edge
				if (t < tL) {       // new min tL
					tL = t;
					if (tL < tE) {  // S leaves before entering polygon
						return -9999;
					}
				}
			}
		}
		
		// tE <= tL implies that there is a valid intersection subsegment
		IS.S() = S.S() + tE * dS;   // = P(tE) = point where S enters polygon
		IS.E() = S.S() + tL * dS;   // = P(tL) = point where S leaves polygon
		
		VEC2D<T> IV = IS.E() - IS.S();
		
		return IV.length();
	}
	
	
	
//	//
//	// TEST main code
//	//
//	int main(int argc, const char * argv[]) {
//		// Input : Rectangle & Segment
//		RECT2D<double> rect(POINT2D<double>(0,0), POINT2D<double>(8,0), POINT2D<double>(8,3), POINT2D<double>(0,3));
//		//	SEGMENT2D<double> S(POINT2D<double>(6, -2), POINT2D<double>(-3,4));
//		SEGMENT2D<double> S(POINT2D<double>(6, -2), POINT2D<double>(0,-2));
//		// Output : Segment
//		SEGMENT2D<double> IS;
//		
//		double intersect_len = intersect::RectangleSegmentIntersection(S, rect, IS);
//		
//		cout<<"intersection length = "<<intersect_len<<endl;
//		cout<<"IS.start = "; IS.S().Print();
//		cout<<"IS.end   = "; IS.E().Print();
//		
//		
//		return 0;
//	}
	
	
} // namespace


#endif	// INTERSECT_H_INCLUDED
