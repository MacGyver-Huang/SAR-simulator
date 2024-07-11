//
//  cad.h
//  PASSEDv4
//
//  Created by Steve Chiang on 2022/11/09.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//
#ifndef cad_h
#define cad_h


#include "assimp/Importer.hpp"      // C++ importer interface
#include "assimp/scene.h"           // Output data structure
#include "assimp/postprocess.h"     // Post processing flags
#include "tinyxml.h"				// XML parser
#include <basic/vec.h>
#include <list/vertexlist.h>
#include <list/connectlist.h>
#include <list/polygonlist.h>
#include <bvh/triangle.h>
#include <vector>
#include <unordered_map>


using namespace vec;
using namespace std;

class VEc{
public:
	VEc(){};
	VEc(const VEC<float>& V){
		v = V;
	}

	const VEC<float> &getV() const {
		return v;
	}

	void setV(const VEC<float> &V) {
		v = V;
	}

	bool operator==(const VEc& other) const{
//		return false;
		return _Distance(v, other.v) < 1e-8;
	}
private:
	double _Distance(const VEC<float>& v1, const VEC<float>& v2) const {
		return sqrt(Square(v1.x()-v2.x()) + Square(v1.y()-v2.y()) + Square(v1.z()-v2.z()));
	}
public:
	VEC<float> v;
};

struct _Hash{
	size_t operator()(const VEc& v) const {
		return std::hash<float>{}(v.v.x() + v.v.y() + v.v.z()); // + is commutative
	};
};

namespace cad {
	//+---------------------------+
	//|     MaterialRGB class     |
	//+---------------------------+
	template<typename T>
	class MATERIALRGB {
	public:
		/**
		 * Default constructor
		 */
		MATERIALRGB(){};
		/**
		 * Constructor
		 */
		MATERIALRGB(const string name, const VEC<T>& RGB){
			_name = name;
			_RGB = RGB;
		}

		MATERIALRGB(const MATERIALRGB<T>& materialrgb) {
			_name = materialrgb.name();
			_RGB  = materialrgb.RGB();
		}

//		MATERIALRGB(MATERIALRGB<T> materialrgb) {
//			_name = materialrgb.name();
//			_RGB  = materialrgb.RGB();
//		}

//		MATERIALRGB(const MATERIALRGB<T> materialrgb) {
//			_name = materialrgb.name();
//			_RGB  = materialrgb.RGB();
//		}

		/**
		 * Get material name (editable)
		 * @return Return material name string
		 */
		string& name(){ return _name; }
		/**
		 * Get material name
		 * @return Return material name string
		 */
		const string& name()const{ return _name; }
		/**
		 * Get RGB color vector (editable)
		 * @return Return a RGB color vector in VEC<T>
		 */
		VEC<T>& RGB(){ return _RGB; }
		/**
		 * Get RGB color vector
		 * @return Return a RGB color vector in VEC<T>
		 */
		const VEC<T>& RGB()const{ return _RGB; }
	private:
		string _name;	// Material name string
		VEC<T> _RGB;	// Material RGB color vector
	};
	//+---------------------------+
	//|         CAD class         |
	//+---------------------------+
	template<typename T>
	class CAD3DX {
	public:
		/**
		 * Default constructor
		 */
		CAD3DX():_IsLoad(false){};
		/**
		 * Constructor with input file. After using this constructor the memeber variables, PL, CL
		 * and VL will be automatically generated.
		 * @param[in]   file   (string) File name
		 */
		CAD3DX(const string File, const MaterialDB& MatDB, const string& Unit = "m", const bool IsCorrect=false){
			_file = File;
			_IsLoad = Load(MatDB, Unit);
//			_Mat.reserve(10000);
//			_MatIdx.reserve(1000000);
			// check
			long MissMatchNumber = CheckMaterialDBBoundary(MatDB, IsCorrect);
			cout<<"+--------------------------+"<<endl;
			cout<<"|       CAD 3D Model       |"<<endl;
			cout<<"+--------------------------+"<<endl;
			if(MissMatchNumber == 0){
				cout<<"No mismatch::All CAD colors(size="<<_MatIdx.size()<<") match the Material Table(size="<<MatDB.Mat.size()<<")"<<endl;
			}else{
				double MissMatchPercentage = (double)MissMatchNumber/(double)_MatIdx.size()*100;
				cout<<"Total = "<<_MatIdx.size()<<", MissMatchNumber = "<<MissMatchNumber<<"("<<MissMatchPercentage<<"%)"<<endl;
				// Show message
				if(IsCorrect){
					cout<<"CAD3DX::WARRNING: All out of boundary MatIdx in MatDB will be replaced to -1(PEC)"<<endl;
				}else{
					cout<<"CAD3DX::ERROR: There are out of boundary MatIdx, MissMatchNumber > 0 without any correction."<<endl;
					cout<<"              If You want to ignore this ERROR, some process you can do:"<<endl;
					cout<<"              1. Checking all index is within the Matrial database(MatDB)"<<endl;
					cout<<"              2. Giving the 3rd arrgument in CAD3DX() constructor to be TRUE,"<<endl;
					cout<<"                 All out of boundary MatIdx in MatDB will be replaced to -1."<<endl;
					exit(EXIT_FAILURE);
				}
			}
		}
		/**
		 * Get vertice index by giving polygon index
		 * @param[in]   idx_poly   Polygon index
		 * @return Return the 1st vertice index(iv0)
		 */
		VEC<T>& getV0(const size_t idx_poly){
			return _VL[ _PL[idx_poly].IV0() ];
		}
		/**
		 * Get vertice index by giving polygon index
		 * @param[in]   idx_poly   Polygon index
		 * @return Return the 2nd vertice index(iv1)
		 */
		VEC<T>& getV1(const size_t idx_poly){ return _VL[ _PL[idx_poly].IV1() ]; }
		/**
		 * Get vertice index by giving polygon index
		 * @param[in]   idx_poly   Polygon index
		 * @return Return the 3rd vertice index(iv2)
		 */
		VEC<T>& getV2(const size_t idx_poly){ return _VL[ _PL[idx_poly].IV2() ]; }
		/**
		 * Get Vertex list (editable)
		 */
		VertexList<T>& VL(){ return _VL; }
		/**
		 * Get Vertex list
		 */
		const VertexList<T>& VL()const{ return _VL; }
		/**
		 * Get Connect list (editable)
		 */
		ConList& CL(){ return _CL; }
		/**
		 * Get Connect list
		 */
		const ConList& CL()const{ return _CL; }
		/**
		 * Get Polygon list (editable)
		 */
		PolygonList& PL(){ return _PL; }
		/**
		 * Get Polygon list
		 */
		const PolygonList& PL()const{ return _PL; }
		/**
		 * Get the size of vertex list
		 */
		size_t size_VL(){ return _VL.size(); }
		/**
		 * Get the size of connect list
		 */
		size_t size_CL(){ return _CL.size(); }
		/**
		 * Get the size of polygon list
		 */
		size_t size_PL(){ return _PL.size(); }
		// Misc.
		void GetCOLLADANameScale(const string filename, string& Unit, double& Scale){
			// Read *.dae with XML format
			TiXmlDocument doc;
			int valid = doc.LoadFile(filename.c_str());

			Unit = "";
			Scale = 1.0;

			if(valid == 1){
				TiXmlElement *L1, *L2, *L3;
				L1 = doc.FirstChildElement("COLLADA");
				if(L1 == NULL){
					cerr<<"ERROR::GetCOLLADANameScale:This is not a COLLADA(*.dae) file!"<<endl;
					exit(EXIT_FAILURE);
				}
				L2 = L1->FirstChildElement("asset");
				L3 = L2->FirstChildElement("unit");
				Unit  = L3->Attribute("name");
				Scale = str2num<double>(L3->Attribute("meter"));
			}
		}
		double GetMaxX(){
			double maxval = -1E+10;
			VEC<double> v0, v1, v2;
			for(size_t i=0;i<_PL.size();++i){
				v0 = getV0(i);
				v1 = getV1(i);;
				v2 = getV2(i);;
				if(v0.x() > maxval){ maxval = v0.x(); }
				if(v1.x() > maxval){ maxval = v1.x(); }
				if(v2.x() > maxval){ maxval = v2.x(); }
			}
			return maxval;
		}
		double GetMaxY(){
			double maxval = -1E+10;
			VEC<double> v0, v1, v2;
			for(size_t i=0;i<_PL.size();++i){
				v0 = getV0(i);
				v1 = getV1(i);;
				v2 = getV2(i);;
				if(v0.y() > maxval){ maxval = v0.y(); }
				if(v1.y() > maxval){ maxval = v1.y(); }
				if(v2.y() > maxval){ maxval = v2.y(); }
			}
			return maxval;
		}
		double GetMaxZ(){
			double maxval = -1E+10;
			VEC<double> v0, v1, v2;
			for(size_t i=0;i<_PL.size();++i){
				v0 = getV0(i);
				v1 = getV1(i);;
				v2 = getV2(i);;
				if(v0.z() > maxval){ maxval = v0.z(); }
				if(v1.z() > maxval){ maxval = v1.z(); }
				if(v2.z() > maxval){ maxval = v2.z(); }
			}
			return maxval;
		}
		double GetMinX(){
			double minval = 1E+10;
			VEC<double> v0, v1, v2;
			for(size_t i=0;i<_PL.size();++i){
				v0 = getV0(i);
				v1 = getV1(i);;
				v2 = getV2(i);;
				if(v0.x() < minval){ minval = v0.x(); }
				if(v1.x() < minval){ minval = v1.x(); }
				if(v2.x() < minval){ minval = v2.x(); }
			}
			return minval;
		}
		double GetMinY(){
			double minval = 1E+10;
			VEC<double> v0, v1, v2;
			for(size_t i=0;i<_PL.size();++i){
				v0 = getV0(i);
				v1 = getV1(i);;
				v2 = getV2(i);;
				if(v0.y() < minval){ minval = v0.y(); }
				if(v1.y() < minval){ minval = v1.y(); }
				if(v2.y() < minval){ minval = v2.y(); }
			}
			return minval;
		}
		double GetMinZ(){
			double minval = 1E+10;
			VEC<double> v0, v1, v2;
			for(size_t i=0;i<_PL.size();++i){
				v0 = getV0(i);
				v1 = getV1(i);;
				v2 = getV2(i);;
				if(v0.z() < minval){ minval = v0.z(); }
				if(v1.z() < minval){ minval = v1.z(); }
				if(v2.z() < minval){ minval = v2.z(); }
			}
			return minval;
		}
		// Print
		void PrintFace(){
			cout<<"+------------+"<<endl;
			cout<<"|    Face    |"<<endl;
			cout<<"+------------+"<<endl;
			VEC<double> v0, v1, v2;
			for(size_t i=0;i<_PL.size();++i){
				v0 = getV0(i);
				v1 = getV1(i);;
				v2 = getV2(i);;
				cout<<"+"<<endl;
				cout<<"Face "<<i<<endl;
				v0.Print();
				v1.Print();
				v2.Print();
				cout<<"Material Index = "<<_MatIdx[i]<<endl;
			}
		}
		void PrintN(){
			cout<<"+------------+"<<endl;
			cout<<"|   Normal   |"<<endl;
			cout<<"+------------+"<<endl;
			VEC<double> v0, v1, v2;
			for(size_t i=0;i<_PL.size();++i){
				v0 = getV0(i);
				v1 = getV1(i);;
				v2 = getV2(i);;
				VEC<double> e01 = v1 - v0;
				VEC<double> e02 = v2 - v0;
				double angle  = abs(vec::angle(e01, e02));
				double Area   = e01.abs()*sin(angle)*e02.abs()/2.;
				VEC<double> N = Unit(cross(e01, e02));
				if(Area > 0.4){
					cout<<"+"<<endl;
					cout<<"Normal "<<i<<", Area = "<<Area<<endl;
					N.Print();
					v0.Print();
					v1.Print();
					v2.Print();
				}
			}
		}
		void PrintMaterial(){
			cout<<"+------------+"<<endl;
			cout<<"|  Material  |"<<endl;
			cout<<"+------------+"<<endl;
			for(unsigned long i=0;i<_Mat.size();++i){
				cout<<"+"<<endl;
				cout<<"Material #"<<i<<endl;
				cout<<"name  = "<<_Mat[i].name()<<endl;
				cout<<"Color = ["<<_Mat[i].RGB()[0]<<","<<_Mat[i].RGB()[1]<<","<<_Mat[i].RGB()[2]<<"]"<<endl;
			}
		}
		/**
		 * Display all memeber variables on console
		 */
		void Print(){
			cout<<"+----------------------------+"<<endl;
			cout<<"|       Summary of CAD       |"<<endl;
			cout<<"+----------------------------+"<<endl;
			_VL.Print();
			_CL.Print();
			_PL.Print();
		}
		/**
		 * Convert the CAD class to object (contains only the vertex)
		 * @remark This the 1st step to import to the BVH class
		 */
		vector<Obj*> Convert2Obj(){
			vector<Obj*> obj;
			obj.reserve( _PL.size() );
			double ea[3];
			long idx_poly_near[3];
			for(size_t i=0;i<_PL.size();++i){
				// Get 3 edge angle
				WedgeAngle(i, ea, idx_poly_near);
				// Add to object
				obj.push_back( new TRI<float>(getV0(i), getV1(i), getV2(i), ea, i, idx_poly_near) );
			}
			return obj;
		}
		/**
		 * Find the normal vector by giving a polygon index
		 * @param[in]   idx_poly   Polygon index
		 * @return Return a VEC<T> that contains normal vector
		 */
		VEC<T> GetNormal(const size_t idx_poly){
			// Prepare for all vertex (index)
			VEC<T> v0 = getV0(idx_poly);
			VEC<T> v1 = getV1(idx_poly);
			VEC<T> v2 = getV2(idx_poly);
			return Unit(cross(v1 - v0, v2 - v0));
		}
		/**
		 * Get triangle by giving idx_poly
		 * @param[in]   idx_poly   Polygon index
		 * @return Return a TRIA<T> that contains triangle(TRIA) class
		 */
		TRI<T> GetTriangle(const size_t idx_poly){
			// Prepare for all vertex (index)
			VEC<T> v0 = getV0(idx_poly);
			VEC<T> v1 = getV1(idx_poly);
			VEC<T> v2 = getV2(idx_poly);

			T ea[3];
			long idx_poly_near[3];

			WedgeAngle(idx_poly, ea, idx_poly_near);
			return TRI<T>(v0, v1, v2, ea, idx_poly, idx_poly_near);
		}
	private:
		bool Load(const MaterialDB& MatDB, const string& unit = "m"){
			// Create an instance of the Importer class
			Assimp::Importer importer;
			// And have it read the given file with some example postprocessing
			// Usually - if speed is not the most important aspect for you - you'll
			// propably to request more postprocessing than we do in this example.
			const aiScene* scene = importer.ReadFile( _file.c_str(), aiProcess_Triangulate | aiProcess_OptimizeGraph );
			
			// Detect file type & Units
			// Parse the unit by the tinyxml library
			// Because the assimp cannot detect the units of the 3DS & COLLADA
			double scale = 1.0;
			string type  = _file.substr(_file.find_last_of(".") + 1);

			if(type == "3ds"){
				cout<<"Input CAD type = 3DS(*.3ds)"<<endl;
				string UNIT = StrUppercase(unit);
				if(UNIT == "10M"){  scale = 10.0; }		// 10 meter (after blender remesh)
				if(UNIT == "M"){  scale = 1.0; }		// meter (e.g T72 MSTAR) (default)
				if(UNIT == "MM"){ scale = 0.001; }		// mini meter (e.g M60)
				if(UNIT == "CM"){ scale = 0.01; }		// centi meter
				if(UNIT == "INCH"){ scale = 0.0254; }	// inch
			}else if(type == "dae"){
				cout<<"Input CAD type = COLLADA(*.dae)"<<endl;
				string UnitDae;
				GetCOLLADANameScale(_file, UnitDae, scale);
				string UNIT = StrUppercase(unit);
//				UNIT = "INCH";
				if(UNIT == "10M"){  scale = 10.0; }		// 10 meter (after blender remesh)
				if(UNIT == "M"){  scale = 1.0; }		// meter (e.g T72 MSTAR) (default)
				if(UNIT == "MM"){ scale = 0.001; }		// mini meter (e.g M60)
				if(UNIT == "CM"){ scale = 0.01; }		// centi meter
				if(UNIT == "INCH"){ scale = 0.0254; }	// inch
			}else if(type == "stl"){
				cout<<"Input CAD type = STL(*.stl)"<<endl;
				string UNIT = StrUppercase(unit);
				if(UNIT == "10M"){  scale = 10.0; }		// 10 meter (after blender remesh)
				if(UNIT == "M"){  scale = 1.0; }		// meter (e.g T72 MSTAR) (default)
				if(UNIT == "MM"){ scale = 0.001; }		// mini meter (e.g M60)
				if(UNIT == "CM"){ scale = 0.01; }		// centi meter
				if(UNIT == "INCH"){ scale = 0.0254; }	// inch
			}else{
				cerr<<"ERROR::The CAD type ("<<type<<") is not supported."<<endl;
				exit(EXIT_FAILURE);
			}
			


			// If the import failed, report it
			if(!scene){
				cout<<importer.GetErrorString()<<endl;
				return false;
			}

			// # Material
			printf("+--------------------------+\n");
			printf("|  Material List from CAD  |\n");
			printf("+--------------------------+\n");
			for(unsigned int i=0;i<scene->mNumMaterials;++i){
				aiMaterial* mat = scene->mMaterials[i];
				aiString name;
				mat->Get(AI_MATKEY_NAME,name);
				aiColor3D C (0.f,0.f,0.f);
				mat->Get(AI_MATKEY_COLOR_DIFFUSE,C);
				_Mat.push_back( MATERIALRGB<float>(aiString2string(name), aiColor3D2VEC(C)) );

				printf("[R,G,B]=[%3d,%3d,%3d], name='%s', RGBIdx=%ld\n",
						int(_Mat[_Mat.size()-1].RGB().x()*255), int(_Mat[_Mat.size()-1].RGB().y()*255), int(_Mat[_Mat.size()-1].RGB().z()*255),
//						int(C.r*255), int(C.g*255), int(C.b*255),
						aiString2string(name).c_str(),
						long( (C.r*255)*256*256 + (C.g*255)*256 + (C.b*255) ) );
			}

			// # Vertex
			printf("+--------------------------+\n");
			printf("|  Vertex List from CAD    |\n");
			printf("+--------------------------+\n");
			// Allocate set data structure
			unordered_map<VEc, size_t, _Hash> unorderedMap;
			// Initial _VL counter
			size_t count = 0;
			// For each Mesh & Face
			for(size_t j=0;j<scene->mNumMeshes;++j){
				printf("Read 3ds file: Mesh = %zu / %u\n", j+1, scene->mNumMeshes);
				// Get Mesh
				aiMesh* M = scene->mMeshes[j];
				// Get previous size
				size_t PrePolygonSize = _PL.size();
				//
				// Store polygon
				//
				for(size_t i=0;i<M->mNumFaces;++i){
					if(mat::Mod((int)(i+1), 5000) == 0 || i == 0) {
						printf("               Face = %zu / %u\n", i+1, M->mNumFaces);
					}
					aiFace F = M->mFaces[i];
					// avoid the other kind of polygon besides the triangle type
					if(F.mNumIndices == 3){
						size_t i0 = F.mIndices[0];	// Vertice index
						size_t i1 = F.mIndices[1];
						size_t i2 = F.mIndices[2];
						// Check and remove duplicate vertex
						aiVector3D v0 = M->mVertices[i0];	// Get vertice class
						aiVector3D v1 = M->mVertices[i1];
						aiVector3D v2 = M->mVertices[i2];
						// Scaling by Unit
						v0.x = v0.x*scale; v0.y = v0.y*scale; v0.z = v0.z*scale;
						v1.x = v1.x*scale; v1.y = v1.y*scale; v1.z = v1.z*scale;
						v2.x = v2.x*scale; v2.y = v2.y*scale; v2.z = v2.z*scale;
						//+---------------------------------------------------+
						//|    Assign the Material by RGB color in CAD        |
						//+---------------------------------------------------+
						// Find RGBIdx by "name"
						// Replace the "RGBIdx" extract from 3ds with MaterialDB by "name"
						VEC<long> RGB = static_cast<VEC<long> >( _Mat[M->mMaterialIndex].RGB() * 255 );
						long RGBIdx = -1;
						string NameFromCAD = _Mat[M->mMaterialIndex].name();
						string NameFromTab, MatNameInTab;
						// Find the matching remark field between in CAD file and in Table
						// If the name is same, assign RGBIdx by index in Table
						for(size_t k=0;k<MatDB.Mat.size();++k){
							NameFromTab = MatDB.Mat[k].remark();
							// Special for Surface with material (20-j4)
							if(type == "stl"){
								RGBIdx = MatDB.Mat[k].idx();
								MatNameInTab = "MANUAL(STL)";
							} else {
								if( StrUppercase(NameFromTab) == StrUppercase(NameFromCAD) ){
									RGBIdx = MatDB.Mat[k].idx();
									MatNameInTab = NameFromTab;
								}
							}
						}
						// RGBIdx = M->mMaterialIndex;
						// RGBIdx = 0;			// PEC
						// RGBIdx = 16777061;	// Alumina
						// assign RGBIdx
						_MatIdx.push_back(RGBIdx);
						//+---------------------------------------------------+
						//|                START (unordered_map)              |
						//+---------------------------------------------------+
						// Insert unique VEC<float>
						size_t iv0=i0, iv1=i1, iv2=i2;
						VEC<float> V0(v0.x, v0.y, v0.z);
						VEC<float> V1(v1.x, v1.y, v1.z);
						VEC<float> V2(v2.x, v2.y, v2.z);

						// v0
						auto isInsert_v0 = unorderedMap.insert( {VEc(V0), count} );
						if(isInsert_v0.second){
							_VL.push_back( V0 );
							iv0 = _VL.size()-1;
							count++;
						}else{
							iv0 = unorderedMap.find(VEc(V0))->second;
						}
						// v1
						auto isInsert_v1 = unorderedMap.insert( {VEc(V1), count} );
						if(isInsert_v1.second){
							_VL.push_back( V1 );
							iv1 = _VL.size()-1;
							count++;
						}else{
							iv1 = unorderedMap.find(VEc(V1))->second;
						}
						// v2
						auto isInsert_v2 = unorderedMap.insert( {VEc(V2), count} );
						if(isInsert_v2.second){
							_VL.push_back( V2 );
							iv2 = _VL.size()-1;
							count++;
						}else{
							iv2 = unorderedMap.find(VEc(V2))->second;
						}
						//+---------------------------------------------------+
						//|                 END (unordered_map)               |
						//+---------------------------------------------------+
						// Assign to lists
						_PL.push_back(iv0, iv1, iv2);
						_CL.push_back(iv0, i + PrePolygonSize);
						_CL.push_back(iv1, i + PrePolygonSize);
						_CL.push_back(iv2, i + PrePolygonSize);
					}
				}
			}

			// Check the every edge with shared edge or not? Show message in console.
			this->checkIsSharedEdgeForEachTriangle();

			return true;
		}
		void Print3D(const aiVector3D& in, float scale = 1){
			cout<<"["<<in.x*scale<<","<<in.y*scale<<","<<in.z*scale<<"]"<<endl;
		}
		/**
		 * Check the material table boundary
		 */
		long CheckMaterialDBBoundary(const MaterialDB& MatDB, const bool IsCorrect=false){
			long NumError = 0;
			size_t count;
			for(size_t i=0;i<_MatIdx.size();++i){
				count = 0;
				for(size_t j=0;j<MatDB.Mat.size();++j){
					if(_MatIdx[i] == MatDB.Mat[j].idx()){
						_MatIdx[i] = j;
						count++;
					}
//					if(i < 10){
//						printf("[CheckMaterialDBBoundary] i=%d/%d, j=%d/%d, MatIdx[i]=%d, MatDB.Mat[j].idx()=%d\n",
//								i, MatIdx.size(), j, MatDB.Mat.size(), MatIdx[i], MatDB.Mat[j].idx());
//					}
				}
				if(count == 0){
					_MatIdx[i] = -1;	// Unknown (forced to be PEC)
					NumError++;
				}
			}
			return NumError;
		}
		VEC<double> aiVector3D2VEC(const aiVector3D& in){
			return VEC<double>(in.x, in.y, in.z);
		}
		VEC<double> aiColor3D2VEC(const aiColor3D& in){
			return VEC<double>(in.r, in.g, in.b);
		}
		/**
		 * Convert the aiString to std::string
		 */
		string aiString2string(const aiString& in){
			return string(in.C_Str());
		}
		/**
		 * Find the wedge angle in radius by giving polygon index. It will automatically search nearest polygon
		 * and return the 3 edge angle with the order (edge0 : v0->v1, edge1 : v1->v2, edge2 : v2->v0)
		 * @param[in]   idx_poly       Self polygon index
		 * @param[out]  ea             3 wedge angle with the order (edge0 : v0->v1, edge1 : v1->v2, edge2 : v2->v0)
		 * @param[out]  idx_poly_near  Near polygon index with 3 edge ([0]:v0->v1, [1]:v1->v2, [2]:v2->v0)
		 */
		void WedgeAngle(const size_t idx_poly, double ea[3], long idx_poly_near[3]){
			ea[0] = ea[1] = ea[2] = -1;
			// Get this polygon normal vector
			size_t iv0, iv1, iv2;
			VEC<T> n = GetNormal(idx_poly, iv0, iv1, iv2);
			// Get connect list
			LinkedList<size_t> CL0 = _CL[ iv0 ];
			LinkedList<size_t> CL1 = _CL[ iv1 ];
			LinkedList<size_t> CL2 = _CL[ iv2 ];

			// Initialization
			idx_poly_near[0] = -1;
			idx_poly_near[1] = -1;
			idx_poly_near[2] = -1;


			//
			// Edge01 = {v0->v1}
			//
			vector<size_t> com01 = GetCommonElement(CL0, CL1);
			if(com01.size() >= 2){
				idx_poly_near[0] = GetUniq(com01, idx_poly);
				VEC<T> n01 = GetNormal(idx_poly_near[0]);
				// Get angle
				ea[0] = vec::angle(n, n01);
				ea[0] = (IsConvex(idx_poly, idx_poly_near[0]))? def::PI - ea[0] : def::PI + ea[0];
			}
			//
			// Edge12 = {v1->v2}
			//
			vector<size_t> com12 = GetCommonElement(CL1, CL2);
			if(com12.size() >= 2){
				idx_poly_near[1] = GetUniq(com12, idx_poly);
				VEC<T> n12 = GetNormal(idx_poly_near[1]);
				ea[1] = vec::angle(n, n12);
				ea[1] = (IsConvex(idx_poly, idx_poly_near[1]))? def::PI - ea[1] : def::PI + ea[1];
			}
			//
			// Edge20 = {v2->v0}
			//
			vector<size_t> com20 = GetCommonElement(CL2, CL0);
			if(com20.size() >= 2){
				idx_poly_near[2] = GetUniq(com20, idx_poly);
				VEC<T> n20 = GetNormal(idx_poly_near[2]);
				ea[2] = vec::angle(n, n20);
				ea[2] = (IsConvex(idx_poly, idx_poly_near[2]))? def::PI - ea[2] : def::PI + ea[2];
			}
		}
		/**
		 * Find the normal vector by giving polygon index and return the 3 vertex index
		 * @param[in]   idx_poly   Polygon index
		 * @param[out]  iv0        1st vertice index
		 * @param[out]  iv1        2nd vertice index
		 * @param[out]  iv2        3rd vertice index
		 * @return Return a VEC<T> that contains normal vector
		 */
		VEC<T> GetNormal(const size_t idx_poly, size_t& iv0, size_t& iv1, size_t& iv2){
			// Prepare for all vertex (index)
			iv0 = _PL[idx_poly].IV0();
			iv1 = _PL[idx_poly].IV1();
			iv2 = _PL[idx_poly].IV2();
			VEC<T> v0 = _VL[ iv0 ];
			VEC<T> v1 = _VL[ iv1 ];
			VEC<T> v2 = _VL[ iv2 ];
			return Unit(cross(v1 - v0, v2 - v0));
		}
		/**
		 * Find the common elements within two List
		 * @param[in]   CL0   1st List
		 * @param[in]   CL1   2nd List
		 * @return Return a vector that contains the common elements between CL0 and CL1
		 */
		vector<size_t> GetCommonElement(LinkedList<size_t>& CL0, LinkedList<size_t>& CL1){
			vector<size_t> com;
			com.reserve(2);
			ListNode<size_t>* cur0;
			ListNode<size_t>* cur1;
			size_t dat0, dat1;
			bool IsOver2 = false;
			
			// Prepare shift to first
			cur0 = CL0.getFirstPtr();
			while (cur0 != NULL) {
				// get data
				dat0 = cur0->getData();
				// Prepare shift to first
				cur1 = CL1.getFirstPtr();
				while (cur1 != NULL) {
					// get data
					dat1 = cur1->getData();
					// Compare
					if(dat0 == dat1){
						com.push_back(cur1->getData());
						// if meet to 2 elements break and resturn
						if(com.size() == 2){
							IsOver2 = true;
							break;
						}
					}
					// Move to next
					cur1 = cur1->getNextPtr();
				}
				if(IsOver2){ break; }
				// Move to next
				cur0 = cur0->getNextPtr();
			}
			
			return com;
		}
		/**
		 * Find the 1st element that is not as same as input value "except"
		 * @param[in]   all     Input vector
		 * @param[in]   except  The value needs to be ignore
		 * @return Return the 1st element value that is not as same as "except"
		 */
		size_t GetUniq(const vector<size_t>& all, const size_t except){
			size_t res;
			for(size_t i=0;i<all.size();++i){
				if(all[i] != except){
					res = all[i];
					break;
				}
			}
			return res;
		}
		/**
		 * Find the two non-shared vertex
		 * @param[in]   iv0   [x] 1st vextice relatived to the polygon index vector
		 * @param[in]   iv1   [x] 2nd vextice relatived to the polygon index vector
		 * @param[out]  iPa   [x] 1st vertice belong to iv0 but not on the shared edge
		 * @param[out]  iPb   [x] 2nd vertice belong to iv0 but not on the shared edge
		 */
		void NonsharedVertexIdx(const vector<size_t> iv0, const vector<size_t> iv1, size_t& iPa, size_t& iPb){
			// Get iPa
			for(int i=0;i<3;++i){
				if(iv0[i] != iv1[0] && iv0[i] != iv1[1] && iv0[i] != iv1[2]){
					iPa = iv0[i];
				}
			}
			// Get iPb
			for(int i=0;i<3;++i){
				if(iv1[i] != iv0[0] && iv1[i] != iv0[1] && iv1[i] != iv0[2]){
					iPb = iv1[i];
				}
			}
		}
		/**
		 * Give two triangle, determine this is Converx(true) or Concave(false)?
		 * @param[in]   idx_poly       [x] Triangle index
		 * @param[in]   idx_next_poly  [x] Triangle index next to the 1st triangle
		 * @return Return the boolean that it is Converx(true) or Concave(false)
		 * @ref https://www.gamedev.net/forums/topic/508442-determining-whether-an-edge-on-a-mesh-is-convex-or-concave/
		 */
		bool IsConvex(const size_t idx_poly, const size_t idx_next_poly){
			// Find non-shared vertice on two triangle
			size_t iPa, iPb;
			NonsharedVertexIdx( _PL[idx_poly].IV(), _PL[idx_next_poly].IV(), iPa, iPb);
			VEC<T> Pa = _VL[iPa];
			VEC<T> Pb = _VL[iPb];
			TRI<T> Tria( getV0(idx_poly), getV1(idx_poly), getV2(idx_poly) );
			TRI<T> Trib( getV0(idx_next_poly), getV1(idx_next_poly), getV2(idx_next_poly) );
			return IsConvex(Tria, Pa, Trib, Pb);
		}
		/**
		 * Give two triangle, determine this is Converx(true) or Concave(false)?
		 * @param[in]   t1   (TRI) 1st Trinagle
		 * @param[in]   P1   [m,m,m] Point belong to t1 but not on the shared edge
		 * @param[in]   t2   (TRI) 2nd Trinagle
		 * @param[in]   P2   [m,m,m] Point belong to t2 but not on the shared edge
		 * @return Return the boolean that it is Converx(true) or Concave(false)
		 * @ref https://www.gamedev.net/forums/topic/508442-determining-whether-an-edge-on-a-mesh-is-convex-or-concave/
		 */
		bool IsConvex(const TRI<T>& t1, const VEC<T>& P1, const TRI<T>& t2, const VEC<T>& P2){
			VEC<T> n1 = t1.getNormal();
			return (dot(P2 - P1, n1) <= 0);
		}
		/**
		 * Get distance from to kinds of 3D VECTOR class
		 * @param[in]   a   (aiVector) aiVector class vector
		 * @param[in]   b   (VEC<T>) VEC<T> class vector
		 * @return Return the distance value
		 */
		float Dist(const aiVector3D& a, const VEC<T>& b){
			return sqrt(Square(a.x - b.x()) + Square(a.y - b.y()) + Square(a.z - b.z()));
		}
		/**
		 * Self convert to uppercase
		 * @param[in]   in_out   (string) input string
		 */
		void toUpper(string& in_out){
			std::transform(in_out.begin(), in_out.end(),in_out.begin(), ::toupper);
		}
		/**
		 * Get suffix of string
		 * @param[in]   str   (string) input string
		 */
		string getSuffix(const string& str){
			D1<string> sub = StrSplit(str, '.');
			string suffix = sub[sub.GetNum()-1];
			toUpper(suffix);
			
			return suffix;
		}
		/**
		 * Check the every edge with shared edge or not?
		 */
		void checkIsSharedEdgeForEachTriangle(){
			TRI<double> tri;
			size_t count = 0;

			for (size_t ii = 0; ii < _PL.size(); ++ii) {
				tri = this->GetTriangle(ii);
				long near0 = tri.IDX_Near(0);
				long near1 = tri.IDX_Near(1);
				long near2 = tri.IDX_Near(2);
				if(near0 < 0 || near1 < 0 || near2 < 0){
					count++;
				}
			}

			if(count > 0){
				cout << "WARNING::There are "<<count<<" edges without a shared edge."<<endl;
				cout<<"         There are "<<_PL.size()*3<<" edges with "<<_PL.size()<<" triangles."<<endl;
				cout<<"         Those non-shared edge will not be calculated."<<endl;
				cout<<"         Please check the CAD file is two side."<<endl<<endl;
			}else{
				cout<<"MESSAGE::All edges have at least one shared edge."<<endl<<endl;
			}
		}
	private:
		VertexList<T> _VL;					// Vertex List (store all vertex position, VEC)
		ConList       _CL;					// Connect List (store the used polygon index. The size is as same as _VL)
		PolygonList   _PL;					// Polygon List (store all vertex index by each polygon)
		bool _IsLoad;						// Load data or not?
		string _file;						// File name
		vector<long> _MatIdx;				// Material index
		vector<MATERIALRGB<float> > _Mat;	// Material : [(#material), VEC]
	};
}

#endif /* cad_h */
