import React from 'react';

export const metadata = {
    title: "User Guide",
    description: "Here the users will find the user guide so that the users can understand how things work on this website"
}

export default function GuideLayout({children}) {
    return (
        <div className="">
            {children}
        </div>
    )
}
