//
//  new_dir.h
//  PhysicalOptics14
//
//  Created by Steve Chiang on 5/12/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef new_dir_h
#define new_dir_h

#include <errno.h>
//#include <stdlib.h>
//#include <string.h>
#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#pragma warning (disable : 4996)
#else
#include <dirent.h>
#include <sys/stat.h>
#endif


#include <string>
#include <iostream>
#include <iomanip>
#include "d1.h"


using namespace std;
using namespace d1;


namespace new_dir {
	/*
	 Copyright (c) 2013-2014, Cong Xu, Baudouin Feildel
	 All rights reserved.
	 
	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:
	 
	 1. Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 2. Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 
	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	 */
	
	/* types */
	
#define _TINYDIR_PATH_MAX 4096
#ifdef _MSC_VER
	/* extra chars for the "\\*" mask */
#define _TINYDIR_PATH_EXTRA 2
#else
#define _TINYDIR_PATH_EXTRA 0
#endif
#define _TINYDIR_FILENAME_MAX 256
	
#ifdef _MSC_VER
#define _TINYDIR_FUNC static __inline
#else
#define _TINYDIR_FUNC static __inline__
#endif
	
	typedef struct
	{
		char path[_TINYDIR_PATH_MAX];
		char name[_TINYDIR_FILENAME_MAX];
		char *extension;
		int is_dir;
		int is_reg;
		
#ifdef _MSC_VER
#else
		struct stat _s;
#endif
	} tinydir_file;
	
	typedef struct
	{
		char path[_TINYDIR_PATH_MAX];
		int has_next;
		size_t n_files;
		
		tinydir_file *_files;
#ifdef _MSC_VER
		HANDLE _h;
		WIN32_FIND_DATA _f;
#else
		DIR *_d;
		struct dirent *_e;
#endif
	} tinydir_dir;
	
	
	/* declarations */
	
	_TINYDIR_FUNC
	int tinydir_open(tinydir_dir *dir, const char *path);
	_TINYDIR_FUNC
	int tinydir_open_sorted(tinydir_dir *dir, const char *path);
	_TINYDIR_FUNC
	void tinydir_close(tinydir_dir *dir);
	
	_TINYDIR_FUNC
	int tinydir_next(tinydir_dir *dir);
	_TINYDIR_FUNC
	int tinydir_readfile(const tinydir_dir *dir, tinydir_file *file);
	_TINYDIR_FUNC
	int tinydir_readfile_n(const tinydir_dir *dir, tinydir_file *file, size_t i);
	_TINYDIR_FUNC
	int tinydir_open_subdir_n(tinydir_dir *dir, size_t i);
	
	_TINYDIR_FUNC
	void _tinydir_get_ext(tinydir_file *file);
	_TINYDIR_FUNC
	int _tinydir_file_cmp(const void *a, const void *b);
	
	
	/* definitions*/
	
	_TINYDIR_FUNC
	int tinydir_open(tinydir_dir *dir, const char *path)
	{
		if (dir == NULL || path == NULL || strlen(path) == 0)
		{
			errno = EINVAL;
			return -1;
		}
		if (strlen(path) + _TINYDIR_PATH_EXTRA >= _TINYDIR_PATH_MAX)
		{
			errno = ENAMETOOLONG;
			return -1;
		}
		
		/* initialise dir */
		dir->_files = NULL;
#ifdef _MSC_VER
		dir->_h = INVALID_HANDLE_VALUE;
#else
		dir->_d = NULL;
#endif
		tinydir_close(dir);
		
		strcpy(dir->path, path);
#ifdef _MSC_VER
		strcat(dir->path, "\\*");
		dir->_h = FindFirstFile(dir->path, &dir->_f);
		dir->path[strlen(dir->path) - 2] = '\0';
		if (dir->_h == INVALID_HANDLE_VALUE)
#else
			dir->_d = opendir(path);
		if (dir->_d == NULL)
#endif
		{
			errno = ENOENT;
			goto bail;
		}
		
		/* read first file */
		dir->has_next = 1;
#ifndef _MSC_VER
		dir->_e = readdir(dir->_d);
		if (dir->_e == NULL)
		{
			dir->has_next = 0;
		}
#endif
		
		return 0;
		
	bail:
		tinydir_close(dir);
		return -1;
	}
	
	_TINYDIR_FUNC
	int tinydir_open_sorted(tinydir_dir *dir, const char *path)
	{
		/* Count the number of files first, to pre-allocate the files array */
		size_t n_files = 0;
		if (tinydir_open(dir, path) == -1)
		{
			return -1;
		}
		while (dir->has_next)
		{
			n_files++;
			if (tinydir_next(dir) == -1)
			{
				goto bail;
			}
		}
		tinydir_close(dir);
		
		if (tinydir_open(dir, path) == -1)
		{
			return -1;
		}
		
		dir->n_files = 0;
		dir->_files = (tinydir_file *)malloc(sizeof *dir->_files * n_files);
		if (dir->_files == NULL)
		{
			errno = ENOMEM;
			goto bail;
		}
		while (dir->has_next)
		{
			tinydir_file *p_file;
			dir->n_files++;
			
			p_file = &dir->_files[dir->n_files - 1];
			if (tinydir_readfile(dir, p_file) == -1)
			{
				goto bail;
			}
			
			if (tinydir_next(dir) == -1)
			{
				goto bail;
			}
			
			/* Just in case the number of files has changed between the first and
			 second reads, terminate without writing into unallocated memory */
			if (dir->n_files == n_files)
			{
				break;
			}
		}
		
		qsort(dir->_files, dir->n_files, sizeof(tinydir_file), _tinydir_file_cmp);
		
		return 0;
		
	bail:
		tinydir_close(dir);
		return -1;
	}
	
	_TINYDIR_FUNC
	void tinydir_close(tinydir_dir *dir)
	{
		if (dir == NULL)
		{
			return;
		}
		
		memset(dir->path, 0, sizeof(dir->path));
		dir->has_next = 0;
		dir->n_files = 0;
		if (dir->_files != NULL)
		{
			free(dir->_files);
		}
		dir->_files = NULL;
#ifdef _MSC_VER
		if (dir->_h != INVALID_HANDLE_VALUE)
		{
			FindClose(dir->_h);
		}
		dir->_h = INVALID_HANDLE_VALUE;
#else
		if (dir->_d)
		{
			closedir(dir->_d);
		}
		dir->_d = NULL;
		dir->_e = NULL;
#endif
	}
	
	_TINYDIR_FUNC
	int tinydir_next(tinydir_dir *dir)
	{
		if (dir == NULL)
		{
			errno = EINVAL;
			return -1;
		}
		if (!dir->has_next)
		{
			errno = ENOENT;
			return -1;
		}
		
#ifdef _MSC_VER
		if (FindNextFile(dir->_h, &dir->_f) == 0)
#else
			dir->_e = readdir(dir->_d);
		if (dir->_e == NULL)
#endif
		{
			dir->has_next = 0;
#ifdef _MSC_VER
			if (GetLastError() != ERROR_SUCCESS &&
				GetLastError() != ERROR_NO_MORE_FILES)
			{
				tinydir_close(dir);
				errno = EIO;
				return -1;
			}
#endif
		}
		
		return 0;
	}
	
	_TINYDIR_FUNC
	int tinydir_readfile(const tinydir_dir *dir, tinydir_file *file)
	{
		if (dir == NULL || file == NULL)
		{
			errno = EINVAL;
			return -1;
		}
#ifdef _MSC_VER
		if (dir->_h == INVALID_HANDLE_VALUE)
#else
			if (dir->_e == NULL)
#endif
			{
				errno = ENOENT;
				return -1;
			}
		if (strlen(dir->path) +
			strlen(
#ifdef _MSC_VER
				   dir->_f.cFileName
#else
				   dir->_e->d_name
#endif
				   ) + 1 + _TINYDIR_PATH_EXTRA >=
			_TINYDIR_PATH_MAX)
		{
			/* the path for the file will be too long */
			errno = ENAMETOOLONG;
			return -1;
		}
		if (strlen(
#ifdef _MSC_VER
				   dir->_f.cFileName
#else
				   dir->_e->d_name
#endif
				   ) >= _TINYDIR_FILENAME_MAX)
		{
			errno = ENAMETOOLONG;
			return -1;
		}
		
		strcpy(file->path, dir->path);
		strcat(file->path, "/");
		strcpy(file->name,
#ifdef _MSC_VER
			   dir->_f.cFileName
#else
			   dir->_e->d_name
#endif
			   );
		strcat(file->path, file->name);
#ifndef _MSC_VER
		if (stat(file->path, &file->_s) == -1)
		{
			return -1;
		}
#endif
		_tinydir_get_ext(file);
		
		file->is_dir =
#ifdef _MSC_VER
		!!(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
#else
		S_ISDIR(file->_s.st_mode);
#endif
		file->is_reg =
#ifdef _MSC_VER
		!!(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_NORMAL) ||
		(
		 !(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_DEVICE) &&
		 !(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
		 !(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_ENCRYPTED) &&
#ifdef FILE_ATTRIBUTE_INTEGRITY_STREAM
		 !(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_INTEGRITY_STREAM) &&
#endif
#ifdef FILE_ATTRIBUTE_NO_SCRUB_DATA
		 !(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_NO_SCRUB_DATA) &&
#endif
		 !(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_OFFLINE) &&
		 !(dir->_f.dwFileAttributes & FILE_ATTRIBUTE_TEMPORARY));
#else
		S_ISREG(file->_s.st_mode);
#endif
		
		return 0;
	}
	
	_TINYDIR_FUNC
	int tinydir_readfile_n(const tinydir_dir *dir, tinydir_file *file, size_t i)
	{
		if (dir == NULL || file == NULL)
		{
			errno = EINVAL;
			return -1;
		}
		if (i >= dir->n_files)
		{
			errno = ENOENT;
			return -1;
		}
		
		memcpy(file, &dir->_files[i], sizeof(tinydir_file));
		_tinydir_get_ext(file);
		
		return 0;
	}
	
	_TINYDIR_FUNC
	int tinydir_open_subdir_n(tinydir_dir *dir, size_t i)
	{
		char path[_TINYDIR_PATH_MAX];
		if (dir == NULL)
		{
			errno = EINVAL;
			return -1;
		}
		if (i >= dir->n_files || !dir->_files[i].is_dir)
		{
			errno = ENOENT;
			return -1;
		}
		
		strcpy(path, dir->_files[i].path);
		tinydir_close(dir);
		if (tinydir_open_sorted(dir, path) == -1)
		{
			return -1;
		}
		
		return 0;
	}
	
	_TINYDIR_FUNC
	void _tinydir_get_ext(tinydir_file *file)
	{
		char *period = strrchr(file->name, '.');
		if (period == NULL)
		{
			file->extension = &(file->name[strlen(file->name)]);
		}
		else
		{
			file->extension = period + 1;
		}
	}
	
	_TINYDIR_FUNC
	int _tinydir_file_cmp(const void *a, const void *b)
	{
		const tinydir_file *fa = (const tinydir_file *)a;
		const tinydir_file *fb = (const tinydir_file *)b;
		if (fa->is_dir != fb->is_dir)
		{
			return -(fa->is_dir - fb->is_dir);
		}
		return strncmp(fa->name, fb->name, _TINYDIR_FILENAME_MAX);
	}
	
	//
	// Misc.
	//
	void MKDIR(const string& folder_name){
		mkdir(folder_name.c_str(),0777);
	}
	
	void MV(const string& From, const string& To){
		rename(From.c_str(), To.c_str());
	}
	
	
	string GetEnvParam(const string EnvString){
		char* tmp = getenv(EnvString.c_str());
		return string(tmp);
	}
	
	//
	// Dir class
	//
	class Dir{
	public:
		Dir(){};
		Dir(const string DIR){
			// Get all files in single folder
			tinydir_dir dir;
			tinydir_open(&dir, DIR.c_str());
			
//			vector<string> file;
//			vector<string> folder;

			while(dir.has_next){
				tinydir_file File;
				tinydir_readfile(&dir, &File);

				string name = File.name;
				if(File.is_dir){
					if(name != "." && name != ".."){
						folder.push_back(name + "/");
					}
				}else{
					file.push_back(name);
				}

				tinydir_next(&dir);
			}

			tinydir_close(&dir);
		}
		D1<string> GetSuffix(const string suffix){
			vector<string> out;
			// Get *.meta
			for(unsigned int i=0;i<file.size();++i){
				D1<string> tmp = StrSplit(file[i], '.');
				if(tmp[tmp.GetNum()-1] == suffix){
					out.push_back(file[i]);
				}
			}
			// convert to D1
			D1<string> res(out.size());
			for(unsigned long i=0;i<out.size();++i){
				res[i] = out[i];
			}
			return res;
		}
		void Print(){
			cout<<"Files :"<<endl;
			for(unsigned int i=0;i<file.size();++i){
				cout<<"'"<<file[i]<<"'"<<endl;
			}

			cout<<"Folders :"<<endl;
			for(unsigned int i=0;i<folder.size();++i){
				cout<<"'"<<folder[i]<<"'"<<endl;
			}
		}
	public:
		vector<string> file;
		vector<string> folder;
	};

	// Misc. functions
	/**
	 * Search the files within the folder
	 *
	 * @param [in] folder - folder name
	 * @param [in] prefix - prefix
	 * @param [in] suffix - suffix
	 * @param [in] IsFullPath - Is the input folder is full path?
	 * @return If the condition is matched, return a string vector
	 */
	vector<string> SearchFileInPASSEDEchoFolder(const string& folder, const string& prefix, const string& suffix, const bool IsFullPath = false){
		DIR* dir;
		struct dirent* ent;

		vector<string> out;

		if ((dir = opendir(folder.c_str())) != NULL) {
			// print all the files and directories within directory
			while((ent = readdir(dir)) != NULL) {
				string name = ent->d_name;
				D1<string> str_p = StrSplit(name, '_');
				D1<string> str_s = StrSplit(name, '.');
				if(str_p.GetNum() > 1 && str_s.GetNum() > 1){
					if(str_p[0] == prefix && str_s[2] == suffix){
						if(IsFullPath){
							out.push_back(folder + name);
						}else{
							out.push_back(name);
						}
					}
				}
			}
			closedir(dir);
		} else {
			// could not open directory
			cerr<<"ERROR::SearchFileInFolder:Cound not open directory"<<endl;
			exit(EXIT_FAILURE);
		}

		return out;
	}
}



#endif
