import yolo.code_generator.h5_to_nncg as h5_to_nncg
import yolo.code_generator.fix_generated_code as fix_generated_code
import yolo.code_generator.move_cpp_file as move_cpp_file


def main():
    try:
        h5_to_nncg.main()
    except:
        pass
    fix_generated_code.main()
    move_cpp_file.main()

if __name__ == '__main__':
    main()
